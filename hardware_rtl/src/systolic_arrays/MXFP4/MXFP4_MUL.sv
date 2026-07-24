module MXFP4_MUL #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    // E2M1 derived constants:
    //   full_mant = 2 bits (implicit + 1 explicit), unsigned product = 4 bits,
    //   signed product = 5 bits, max shift = 4 (max_exp_a + max_exp_b = 2+2)
    //   PRODUCT_WIDTH = 5 + 4 = 9
    //   FIXED_SCALE = 2 (1 fractional bit per mantissa, 2 total in product)
    , parameter PRODUCT_WIDTH = 9
    , parameter FIXED_SCALE = 2
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)
    , parameter dot_out_bits = PRODUCT_WIDTH + LOG2_GROUP_SIZE
) (
    input logic clk
    , input logic rst_n

    , input logic [shared_exp_bits-1:0] mxfp4_a_exp
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem
    , input logic [shared_exp_bits-1:0] mxfp4_b_exp
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem

    , output logic [shared_exp_bits:0] out_exp
    , output logic signed [dot_out_bits-1:0] out_sum
);

    // E2M1 format constants
    localparam [1:0] FP4_EXP_BIAS = 2'd1;
    localparam FULL_MANT_W = 2;       // implicit bit + 1 explicit mantissa bit
    localparam UNSIGNED_PROD_W = 4;   // 2 * FULL_MANT_W
    localparam SIGNED_PROD_W = 5;     // UNSIGNED_PROD_W + 1

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

    logic [shared_exp_bits:0] combined_exp_pre;
    logic signed [dot_out_bits-1:0] out_sum_pre;

    // Per-element E2M1 decode, multiply, and fixed-point conversion
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            // Extract E2M1 fields: {sign, exp[1:0], mant[0]}
            act_sign[i]     = mxfp4_a_elem[i][3];
            act_exp_raw[i]  = mxfp4_a_elem[i][2:1];
            act_mant_raw[i] = mxfp4_a_elem[i][0];

            wgt_sign[i]     = mxfp4_b_elem[i][3];
            wgt_exp_raw[i]  = mxfp4_b_elem[i][2:1];
            wgt_mant_raw[i] = mxfp4_b_elem[i][0];

            // Exponent: subtract bias; subnormal (exp_raw==0) keeps exponent 0
            act_exp[i] = (act_exp_raw[i] == 2'b00) ? 2'b00 : act_exp_raw[i] - FP4_EXP_BIAS;
            wgt_exp[i] = (wgt_exp_raw[i] == 2'b00) ? 2'b00 : wgt_exp_raw[i] - FP4_EXP_BIAS;

            // Full mantissa: prepend implicit bit (1 for normal, 0 for subnormal)
            act_mant[i] = {(act_exp_raw[i] != 2'b00), act_mant_raw[i]};
            wgt_mant[i] = {(wgt_exp_raw[i] != 2'b00), wgt_mant_raw[i]};

            // Unsigned mantissa product (2-bit x 2-bit = 4-bit)
            prod_raw[i] = act_mant[i] * wgt_mant[i];

            // Signed product (negate if signs differ)
            prod_sgn[i] = (act_sign[i] ^ wgt_sign[i])
                        ? -$signed({1'b0, prod_raw[i]})
                        :  $signed({1'b0, prod_raw[i]});

            // Shift amount: sum of per-element unbiased exponents (0 to 4)
            shamt[i] = {1'b0, act_exp[i]} + {1'b0, wgt_exp[i]};

            // Convert to fixed-point: sign-extend to PRODUCT_WIDTH then left-shift
            prod_fixed[i] = PRODUCT_WIDTH'(signed'(prod_sgn[i])) << shamt[i];
        end
    end

    // Combined shared exponent, subtract FIXED_SCALE so the PE reuses
    // the same normalization logic as MXINT4_PE
    always_comb begin
        combined_exp_pre = (shared_exp_bits + 1)'(mxfp4_a_exp)
                         + (shared_exp_bits + 1)'(mxfp4_b_exp)
                         - (shared_exp_bits + 1)'(FIXED_SCALE);
    end

    // Adder tree with per-stage 1-bit width growth
    genvar s, i;
    generate
        for (s = 0; s <= LOG2_GROUP_SIZE; s++) begin : tree
            localparam int W = PRODUCT_WIDTH + s;
            localparam int N = group_size >> s;
            logic signed [W-1:0] val [N-1:0];

            if (s == 0) begin : init
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = prod_fixed[i];
                end
            end else begin : reduce
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = W'(signed'(tree[s-1].val[2*i]))
                                  + W'(signed'(tree[s-1].val[2*i+1]));
                end
            end
        end
    endgenerate

    assign out_sum_pre = tree[LOG2_GROUP_SIZE].val[0];

    // Register outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_exp <= '0;
            out_sum <= '0;
        end else begin
            out_exp <= combined_exp_pre;
            out_sum <= out_sum_pre;
        end
    end

endmodule
