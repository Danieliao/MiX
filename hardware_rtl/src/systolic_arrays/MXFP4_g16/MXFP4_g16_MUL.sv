// Block-32-PE MXFP4 (4.5b): two 16-element MXFP4 sub-blocks per PE, each KEEPING
// its own E8M0 scale (scale block size stays 16 -> still 4.5b; only the PE width
// is 32). Emits one (exp, sum) per sub-block; the PE normalizes each and sums the
// two sub-blocks with the accumulator in a fused 3-input FP add.
module MXFP4_g16_MUL #(
    parameter group_size = 32
    , parameter sub_group_size = 16      // E8M0 scale block (unchanged)
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter PRODUCT_WIDTH = 9
    , parameter FIXED_SCALE = 2
    , parameter NUM_SUB = group_size / sub_group_size            // 2
    , parameter LOG2_SUB_SIZE = $clog2(sub_group_size)           // 4
    , parameter dot_out_bits = PRODUCT_WIDTH + LOG2_SUB_SIZE     // 13
) (
    input logic clk
    , input logic rst_n

    , input logic [NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_a_exp
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem
    , input logic [NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_b_exp
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem

    , output logic [NUM_SUB-1:0][shared_exp_bits:0] out_exp
    , output logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum
);

    localparam [1:0] FP4_EXP_BIAS = 2'd1;
    localparam FULL_MANT_W = 2;
    localparam UNSIGNED_PROD_W = 4;
    localparam SIGNED_PROD_W = 5;

    logic [group_size-1:0] act_sign, wgt_sign;
    logic [group_size-1:0][1:0] act_exp_raw, wgt_exp_raw;
    logic [group_size-1:0] act_mant_raw, wgt_mant_raw;
    logic [group_size-1:0][1:0] act_exp, wgt_exp;
    logic [group_size-1:0][FULL_MANT_W-1:0] act_mant, wgt_mant;
    logic [group_size-1:0][UNSIGNED_PROD_W-1:0] prod_raw;
    logic signed [group_size-1:0][SIGNED_PROD_W-1:0] prod_sgn;
    logic [group_size-1:0][2:0] shamt;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_fixed;

    logic [NUM_SUB-1:0][shared_exp_bits:0] combined_exp_pre;
    logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum_pre;

    // Per-element E2M1 decode + fixed-point product (identical to MXFP4_MUL)
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            act_sign[i]     = mxfp4_a_elem[i][3];
            act_exp_raw[i]  = mxfp4_a_elem[i][2:1];
            act_mant_raw[i] = mxfp4_a_elem[i][0];

            wgt_sign[i]     = mxfp4_b_elem[i][3];
            wgt_exp_raw[i]  = mxfp4_b_elem[i][2:1];
            wgt_mant_raw[i] = mxfp4_b_elem[i][0];

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
        end

        for (int g = 0; g < NUM_SUB; g++) begin
            combined_exp_pre[g] = (shared_exp_bits + 1)'(mxfp4_a_exp[g])
                                + (shared_exp_bits + 1)'(mxfp4_b_exp[g])
                                - (shared_exp_bits + 1)'(FIXED_SCALE);
        end
    end

    // One adder tree per 16-element sub-block
    genvar g, s, i;
    generate
        for (g = 0; g < NUM_SUB; g++) begin : sub
            for (s = 0; s <= LOG2_SUB_SIZE; s++) begin : tree
                localparam int W = PRODUCT_WIDTH + s;
                localparam int N = sub_group_size >> s;
                logic signed [W-1:0] val [N-1:0];

                if (s == 0) begin : init
                    for (i = 0; i < N; i++) begin : elem
                        assign val[i] = prod_fixed[g * sub_group_size + i];
                    end
                end else begin : reduce
                    for (i = 0; i < N; i++) begin : elem
                        assign val[i] = W'(signed'(sub[g].tree[s-1].val[2*i]))
                                      + W'(signed'(sub[g].tree[s-1].val[2*i+1]));
                    end
                end
            end
            assign out_sum_pre[g] = sub[g].tree[LOG2_SUB_SIZE].val[0];
        end
    endgenerate

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
