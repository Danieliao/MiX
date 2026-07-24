module AMXFP4_MUL #(
    parameter group_size = 32
    , parameter scale_bits = 7          // E5M2 shared scale, UNSIGNED (no sign bit)
    , parameter elem_bits = 4           // E2M1 element
    // E2M1 derived constants (identical to MXFP4_MUL):
    //   full_mant = 2 bits, unsigned product = 4 bits, signed product = 5 bits,
    //   PRODUCT_WIDTH = 5 + 4 = 9, FIXED_SCALE = 2
    , parameter PRODUCT_WIDTH = 9
    , parameter SCALE_EXP_BITS = 5
    , parameter SCALE_MANT_BITS = 2
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)
    , parameter dot_out_bits = PRODUCT_WIDTH + LOG2_GROUP_SIZE
    , parameter scale_exp_sum_bits = SCALE_EXP_BITS + 1
    , parameter scale_mant_product_bits = 2 * (SCALE_MANT_BITS + 1)
) (
    input logic clk
    , input logic rst_n

    // Activation block A: 32 E2M1 elements + two E5M2 scales (positive / negative)
    , input logic [scale_bits-1:0] amxfp4_a_scale_pos
    , input logic [scale_bits-1:0] amxfp4_a_scale_neg
    , input logic [group_size-1:0][elem_bits-1:0] amxfp4_a_elem
    // Weight block B: 32 E2M1 elements + two E5M2 scales
    , input logic [scale_bits-1:0] amxfp4_b_scale_pos
    , input logic [scale_bits-1:0] amxfp4_b_scale_neg
    , input logic [group_size-1:0][elem_bits-1:0] amxfp4_b_elem

    // Four quadrants indexed {PP=0, PN=1, NP=2, NN=3}
    , output logic [3:0][scale_exp_sum_bits-1:0] out_scale_exp_sum
    , output logic signed [3:0][dot_out_bits-1:0] out_sum
    , output logic [3:0][scale_mant_product_bits-1:0] out_scale_mant_product
);

    // E2M1 format constants
    localparam [1:0] FP4_EXP_BIAS = 2'd1;
    localparam FULL_MANT_W = 2;       // implicit bit + 1 explicit mantissa bit
    localparam UNSIGNED_PROD_W = 4;   // 2 * FULL_MANT_W
    localparam SIGNED_PROD_W = 5;     // UNSIGNED_PROD_W + 1
    localparam SCALE_FULL_MANT_W = SCALE_MANT_BITS + 1; // implicit bit + 2 mantissa bits = 3

    // Per-element decode signals
    logic [group_size-1:0] act_sign, wgt_sign;
    logic [group_size-1:0][1:0] act_exp_raw, wgt_exp_raw;
    logic [group_size-1:0] act_mant_raw, wgt_mant_raw;
    logic [group_size-1:0][1:0] act_exp, wgt_exp;
    logic [group_size-1:0][FULL_MANT_W-1:0] act_mant, wgt_mant;
    logic [group_size-1:0][UNSIGNED_PROD_W-1:0] prod_raw;
    logic signed [group_size-1:0][SIGNED_PROD_W-1:0] prod_sgn;
    logic [group_size-1:0][2:0] shamt;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_fixed;

    // Per-element signed product, masked into its quadrant (zero in the other three)
    logic signed [3:0][group_size-1:0][PRODUCT_WIDTH-1:0] prod_quad;

    // E5M2 scale decode (one full mantissa + raw exponent per scale)
    logic [SCALE_EXP_BITS-1:0]  a_pos_exp, a_neg_exp, b_pos_exp, b_neg_exp;
    logic [SCALE_MANT_BITS-1:0] a_pos_mant, a_neg_mant, b_pos_mant, b_neg_mant;
    logic [SCALE_FULL_MANT_W-1:0] a_pos_fm, a_neg_fm, b_pos_fm, b_neg_fm;

    logic [3:0][scale_exp_sum_bits-1:0] scale_exp_sum_pre;
    logic [3:0][scale_mant_product_bits-1:0] scale_mant_product_pre;
    logic signed [3:0][dot_out_bits-1:0] out_sum_pre;

    // Per-element E2M1 decode, multiply, fixed-point conversion, quadrant routing
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            // Extract E2M1 fields: {sign, exp[1:0], mant[0]}
            act_sign[i]     = amxfp4_a_elem[i][3];
            act_exp_raw[i]  = amxfp4_a_elem[i][2:1];
            act_mant_raw[i] = amxfp4_a_elem[i][0];

            wgt_sign[i]     = amxfp4_b_elem[i][3];
            wgt_exp_raw[i]  = amxfp4_b_elem[i][2:1];
            wgt_mant_raw[i] = amxfp4_b_elem[i][0];

            // Exponent: subtract bias; subnormal (exp_raw==0) keeps exponent 0
            act_exp[i] = (act_exp_raw[i] == 2'b00) ? 2'b00 : act_exp_raw[i] - FP4_EXP_BIAS;
            wgt_exp[i] = (wgt_exp_raw[i] == 2'b00) ? 2'b00 : wgt_exp_raw[i] - FP4_EXP_BIAS;

            // Full mantissa: prepend implicit bit (1 for normal, 0 for subnormal)
            act_mant[i] = {(act_exp_raw[i] != 2'b00), act_mant_raw[i]};
            wgt_mant[i] = {(wgt_exp_raw[i] != 2'b00), wgt_mant_raw[i]};

            // Unsigned mantissa product (2-bit x 2-bit = 4-bit)
            prod_raw[i] = act_mant[i] * wgt_mant[i];

            // Signed product (negate if signs differ): carries sign_a ^ sign_b
            prod_sgn[i] = (act_sign[i] ^ wgt_sign[i])
                        ? -$signed({1'b0, prod_raw[i]})
                        :  $signed({1'b0, prod_raw[i]});

            // Shift amount: sum of per-element unbiased exponents (0 to 4)
            shamt[i] = {1'b0, act_exp[i]} + {1'b0, wgt_exp[i]};

            // Convert to fixed-point: sign-extend to PRODUCT_WIDTH then left-shift
            prod_fixed[i] = PRODUCT_WIDTH'(signed'(prod_sgn[i])) << shamt[i];

            // Route the (already correctly signed) product into one of four quadrants
            // selected by (sign_a, sign_b); zero in the other three.
            prod_quad[0][i] = (~act_sign[i] & ~wgt_sign[i]) ? prod_fixed[i] : '0; // PP
            prod_quad[1][i] = (~act_sign[i] &  wgt_sign[i]) ? prod_fixed[i] : '0; // PN
            prod_quad[2][i] = ( act_sign[i] & ~wgt_sign[i]) ? prod_fixed[i] : '0; // NP
            prod_quad[3][i] = ( act_sign[i] &  wgt_sign[i]) ? prod_fixed[i] : '0; // NN
        end

        // E5M2 scale decode: high SCALE_EXP_BITS = exponent, low SCALE_MANT_BITS = mantissa
        a_pos_exp  = amxfp4_a_scale_pos[scale_bits-1:SCALE_MANT_BITS];
        a_pos_mant = amxfp4_a_scale_pos[SCALE_MANT_BITS-1:0];
        a_neg_exp  = amxfp4_a_scale_neg[scale_bits-1:SCALE_MANT_BITS];
        a_neg_mant = amxfp4_a_scale_neg[SCALE_MANT_BITS-1:0];
        b_pos_exp  = amxfp4_b_scale_pos[scale_bits-1:SCALE_MANT_BITS];
        b_pos_mant = amxfp4_b_scale_pos[SCALE_MANT_BITS-1:0];
        b_neg_exp  = amxfp4_b_scale_neg[scale_bits-1:SCALE_MANT_BITS];
        b_neg_mant = amxfp4_b_scale_neg[SCALE_MANT_BITS-1:0];

        // Implicit leading bit: 1 for normal (exp!=0), 0 for subnormal
        a_pos_fm = {(a_pos_exp != '0), a_pos_mant};
        a_neg_fm = {(a_neg_exp != '0), a_neg_mant};
        b_pos_fm = {(b_pos_exp != '0), b_pos_mant};
        b_neg_fm = {(b_neg_exp != '0), b_neg_mant};

        // Four result scales = product of the per-sign scales (exp sum + mant product)
        scale_exp_sum_pre[0] = scale_exp_sum_bits'(a_pos_exp) + scale_exp_sum_bits'(b_pos_exp);
        scale_exp_sum_pre[1] = scale_exp_sum_bits'(a_pos_exp) + scale_exp_sum_bits'(b_neg_exp);
        scale_exp_sum_pre[2] = scale_exp_sum_bits'(a_neg_exp) + scale_exp_sum_bits'(b_pos_exp);
        scale_exp_sum_pre[3] = scale_exp_sum_bits'(a_neg_exp) + scale_exp_sum_bits'(b_neg_exp);

        scale_mant_product_pre[0] = a_pos_fm * b_pos_fm;
        scale_mant_product_pre[1] = a_pos_fm * b_neg_fm;
        scale_mant_product_pre[2] = a_neg_fm * b_pos_fm;
        scale_mant_product_pre[3] = a_neg_fm * b_neg_fm;
    end

    // Four adder trees (one per quadrant), per-stage 1-bit width growth
    genvar q, s, i;
    generate
        for (q = 0; q < 4; q++) begin : gen_quad
            for (s = 0; s <= LOG2_GROUP_SIZE; s++) begin : tree
                localparam int W = PRODUCT_WIDTH + s;
                localparam int N = group_size >> s;
                logic signed [W-1:0] val [N-1:0];

                if (s == 0) begin : init
                    for (i = 0; i < N; i++) begin : elem
                        assign val[i] = signed'(prod_quad[q][i]);
                    end
                end else begin : reduce
                    for (i = 0; i < N; i++) begin : elem
                        assign val[i] = W'(signed'(gen_quad[q].tree[s-1].val[2*i]))
                                      + W'(signed'(gen_quad[q].tree[s-1].val[2*i+1]));
                    end
                end
            end
            assign out_sum_pre[q] = gen_quad[q].tree[LOG2_GROUP_SIZE].val[0];
        end
    endgenerate

    // Register outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_scale_exp_sum      <= '0;
            out_sum                <= '0;
            out_scale_mant_product <= '0;
        end else begin
            out_scale_exp_sum      <= scale_exp_sum_pre;
            out_sum                <= out_sum_pre;
            out_scale_mant_product <= scale_mant_product_pre;
        end
    end

endmodule
