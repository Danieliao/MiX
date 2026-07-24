module MXFP4_PLUS_MUL #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8        // E8M0 shared exponent
    , parameter elem_bits = 4              // E2M1 / E0M3 element
    , parameter bm_idx_bits = $clog2(group_size)
    // E2M1 derived constants (identical to MXFP4_MUL):
    //   full_mant = 2 bits, signed product = 5 bits, PRODUCT_WIDTH = 9, FIXED_SCALE = 2
    , parameter PRODUCT_WIDTH = 9
    , parameter FIXED_SCALE = 2
    , parameter COMBINE_FRAC = 6           // finest fractional base (BM x BM = 3+3 frac bits)
    , parameter BM_FULL_MANT_W = 4         // implicit 1 + 3 mantissa bits (E0M3)
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)
    , parameter dot_out_bits = PRODUCT_WIDTH + LOG2_GROUP_SIZE
    , parameter combine_bits = dot_out_bits + (COMBINE_FRAC - FIXED_SCALE) + 3
    , parameter combine_exp_bits = shared_exp_bits + 1
) (
    input logic clk
    , input logic rst_n

    // Activation block A: E8M0 exp + 32 E2M1 elements (E0M3 at the BM index) + BM index
    , input logic [shared_exp_bits-1:0] a_exp
    , input logic [group_size-1:0][elem_bits-1:0] a_elem
    , input logic [bm_idx_bits-1:0] a_bm_idx
    // Weight block B: same layout
    , input logic [shared_exp_bits-1:0] b_exp
    , input logic [group_size-1:0][elem_bits-1:0] b_elem
    , input logic [bm_idx_bits-1:0] b_bm_idx

    , output logic [combine_exp_bits-1:0] out_exp
    , output logic signed [combine_bits-1:0] out_sum
);

    // E2M1 / E0M3 format constants
    localparam [1:0] FP4_EXP_BIAS = 2'd1;
    localparam FULL_MANT_W = 2;        // E2M1 implicit bit + 1 explicit mantissa bit
    localparam UNSIGNED_PROD_W = 4;
    localparam [1:0] BM_IMPLIED_EXP = 2'd2;          // top binade (floor shared exponent)
    localparam SLOT_PROD_W = 2 * BM_FULL_MANT_W;     // 4x4 = 8
    localparam SLOT_SHIFT_MAX = 4;                   // BM exp(2) + BM exp(2)
    localparam SLOT_ALIGNED_W = SLOT_PROD_W + SLOT_SHIFT_MAX; // 12
    localparam TREE_ALIGN = COMBINE_FRAC - FIXED_SCALE;       // 4

    // ---- Regular E2M1 path (identical to MXFP4_MUL, plus BM-position zero mask) ----
    logic [group_size-1:0] act_sign, wgt_sign;
    logic [group_size-1:0][1:0] act_exp_raw, wgt_exp_raw;
    logic [group_size-1:0] act_mant_raw, wgt_mant_raw;
    logic [group_size-1:0][1:0] act_exp, wgt_exp;
    logic [group_size-1:0][FULL_MANT_W-1:0] act_mant, wgt_mant;
    logic [group_size-1:0][UNSIGNED_PROD_W-1:0] prod_raw;
    logic signed [group_size-1:0][4:0] prod_sgn;
    logic [group_size-1:0][2:0] shamt;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_fixed;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_masked;

    // ---- BM (E0M3) decode and forward & swap ----
    logic [elem_bits-1:0] abm_field, bbm_field, bcross_field, across_field;
    logic abm_sign, bbm_sign, bcross_sign, across_sign;
    logic [BM_FULL_MANT_W-1:0] abm_mant, bbm_mant, bcross_mant, across_mant;
    logic [1:0] bcross_exp_raw, across_exp_raw;
    logic [1:0] bcross_exp, across_exp;
    logic match;

    // slot operands (each: 4-bit mantissa with 3 fractional bits, 2-bit exponent, sign)
    logic [BM_FULL_MANT_W-1:0] s0a_mant, s0b_mant, s1a_mant, s1b_mant;
    logic [1:0] s0a_exp, s0b_exp, s1a_exp, s1b_exp;
    logic s0a_sign, s0b_sign, s1a_sign, s1b_sign;
    logic s1_active;

    logic [SLOT_PROD_W-1:0] s0_prod, s1_prod;
    logic [2:0] s0_shift, s1_shift;
    logic s0_sign, s1_sign;
    logic [SLOT_ALIGNED_W-1:0] s0_shifted, s1_shifted;
    logic signed [combine_bits-1:0] s0_aligned, s1_aligned, bm_combined;

    logic signed [combine_bits-1:0] combined_sum_pre;
    logic [combine_exp_bits-1:0] combined_exp_pre;

    logic signed [dot_out_bits-1:0] tree_sum;

    // Per-element E2M1 decode, multiply, fixed-point conversion, BM-position masking
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            act_sign[i]     = a_elem[i][3];
            act_exp_raw[i]  = a_elem[i][2:1];
            act_mant_raw[i] = a_elem[i][0];
            wgt_sign[i]     = b_elem[i][3];
            wgt_exp_raw[i]  = b_elem[i][2:1];
            wgt_mant_raw[i] = b_elem[i][0];

            act_exp[i] = (act_exp_raw[i] == 2'b00) ? 2'b00 : act_exp_raw[i] - FP4_EXP_BIAS;
            wgt_exp[i] = (wgt_exp_raw[i] == 2'b00) ? 2'b00 : wgt_exp_raw[i] - FP4_EXP_BIAS;

            act_mant[i] = {(act_exp_raw[i] != 2'b00), act_mant_raw[i]};
            wgt_mant[i] = {(wgt_exp_raw[i] != 2'b00), wgt_mant_raw[i]};

            prod_raw[i] = act_mant[i] * wgt_mant[i];

            prod_sgn[i] = (act_sign[i] ^ wgt_sign[i])
                        ? -$signed({1'b0, prod_raw[i]})
                        :  $signed({1'b0, prod_raw[i]});

            shamt[i] = {1'b0, act_exp[i]} + {1'b0, wgt_exp[i]};

            prod_fixed[i] = PRODUCT_WIDTH'(signed'(prod_sgn[i])) << shamt[i];

            // Zero the BM positions: their elements are E0M3 (handled by the BM unit),
            // so they must not be counted at E2M1 precision in the regular tree.
            prod_masked[i] = ((i == a_bm_idx) || (i == b_bm_idx)) ? '0 : prod_fixed[i];
        end

        // ---- Forward & swap: BM operands and cross partners ----
        abm_field    = a_elem[a_bm_idx];
        bbm_field    = b_elem[b_bm_idx];
        bcross_field = b_elem[a_bm_idx];   // B's (regular) element at A's BM index
        across_field = a_elem[b_bm_idx];   // A's (regular) element at B's BM index

        // BM decode (E0M3): implicit leading 1, implied exponent = top binade
        abm_sign = abm_field[3];
        abm_mant = {1'b1, abm_field[2:0]};
        bbm_sign = bbm_field[3];
        bbm_mant = {1'b1, bbm_field[2:0]};

        // Cross partners decoded as E2M1, then promoted to 3 fractional bits ({impl, m, 00})
        bcross_sign    = bcross_field[3];
        bcross_exp_raw = bcross_field[2:1];
        bcross_exp     = (bcross_exp_raw == 2'b00) ? 2'b00 : bcross_exp_raw - FP4_EXP_BIAS;
        bcross_mant    = {(bcross_exp_raw != 2'b00), bcross_field[0], 2'b00};

        across_sign    = across_field[3];
        across_exp_raw = across_field[2:1];
        across_exp     = (across_exp_raw == 2'b00) ? 2'b00 : across_exp_raw - FP4_EXP_BIAS;
        across_mant    = {(across_exp_raw != 2'b00), across_field[0], 2'b00};

        match = (a_bm_idx == b_bm_idx);

        // slot0 = Abm x (Bbm if match else B@ia);  slot1 = (A@ib x Bbm) when !match else 0
        s0a_mant = abm_mant;  s0a_exp = BM_IMPLIED_EXP;            s0a_sign = abm_sign;
        s0b_mant = match ? bbm_mant : bcross_mant;
        s0b_exp  = match ? BM_IMPLIED_EXP : bcross_exp;
        s0b_sign = match ? bbm_sign : bcross_sign;

        s1_active = ~match;
        s1a_mant = across_mant; s1a_exp = across_exp;             s1a_sign = across_sign;
        s1b_mant = bbm_mant;    s1b_exp = BM_IMPLIED_EXP;         s1b_sign = bbm_sign;

        // BM products (4x4 mantissa, frac base 6), aligned by exponent shift
        s0_prod    = s0a_mant * s0b_mant;
        s0_shift   = {1'b0, s0a_exp} + {1'b0, s0b_exp};
        s0_sign    = s0a_sign ^ s0b_sign;
        s0_shifted = SLOT_ALIGNED_W'(s0_prod) << s0_shift;
        s0_aligned = s0_sign ? -$signed({1'b0, s0_shifted}) : $signed({1'b0, s0_shifted});

        s1_prod    = s1a_mant * s1b_mant;
        s1_shift   = {1'b0, s1a_exp} + {1'b0, s1b_exp};
        s1_sign    = s1a_sign ^ s1b_sign;
        s1_shifted = SLOT_ALIGNED_W'(s1_prod) << s1_shift;
        s1_aligned = ~s1_active ? '0
                   : (s1_sign ? -$signed({1'b0, s1_shifted}) : $signed({1'b0, s1_shifted}));

        bm_combined = s0_aligned + s1_aligned;

        // Combine the regular tree result (base 2 -> base 6) with the BM products (base 6)
        combined_sum_pre = (combine_bits'(signed'(tree_sum)) <<< TREE_ALIGN) + bm_combined;

        // Block exponent with the fractional base folded in
        combined_exp_pre = combine_exp_bits'(a_exp) + combine_exp_bits'(b_exp)
                         - combine_exp_bits'(COMBINE_FRAC);
    end

    // Regular adder tree over BM-masked products (per-stage 1-bit width growth)
    genvar s, i;
    generate
        for (s = 0; s <= LOG2_GROUP_SIZE; s++) begin : tree
            localparam int W = PRODUCT_WIDTH + s;
            localparam int N = group_size >> s;
            logic signed [W-1:0] val [N-1:0];

            if (s == 0) begin : init
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = signed'(prod_masked[i]);
                end
            end else begin : reduce
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = W'(signed'(tree[s-1].val[2*i]))
                                  + W'(signed'(tree[s-1].val[2*i+1]));
                end
            end
        end
    endgenerate
    assign tree_sum = tree[LOG2_GROUP_SIZE].val[0];

    // Register outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_exp <= '0;
            out_sum <= '0;
        end else begin
            out_exp <= combined_exp_pre;
            out_sum <= combined_sum_pre;
        end
    end

endmodule
