// Block-32 NVFP4 MUL: processes two independent 16-element NVFP4 sub-blocks,
// each with its own E4M3 scale (NVFP4's native scale granularity is per-16, so a
// 32-wide PE carries two scales per operand). Emits one (exp_sum, sum, mant_product)
// per sub-block; the PE applies each scale and sums the two sub-blocks with the
// accumulator (the "5th adder stage" — done after scaling, since the sub-blocks
// have independent scales).
module NVFP4_MUL #(
    parameter group_size = 32
    , parameter sub_group_size = 16     // NVFP4 native scale block
    , parameter scale_bits = 8          // E4M3 block scale
    , parameter elem_bits = 4           // E2M1 element
    , parameter PRODUCT_WIDTH = 9
    , parameter SCALE_EXP_BITS = 4
    , parameter SCALE_MANT_BITS = 3
    , parameter NUM_SUB = group_size / sub_group_size            // 2
    , parameter LOG2_SUB_SIZE = $clog2(sub_group_size)           // 4
    , parameter dot_out_bits = PRODUCT_WIDTH + LOG2_SUB_SIZE     // 13
    , parameter scale_exp_sum_bits = SCALE_EXP_BITS + 1          // 5
    , parameter scale_mant_product_bits = 2 * (SCALE_MANT_BITS + 1) // 8
) (
    input logic clk
    , input logic rst_n

    , input logic [NUM_SUB-1:0][scale_bits-1:0] nvfp4_a_scale
    , input logic [group_size-1:0][elem_bits-1:0] nvfp4_a_elem
    , input logic [NUM_SUB-1:0][scale_bits-1:0] nvfp4_b_scale
    , input logic [group_size-1:0][elem_bits-1:0] nvfp4_b_elem

    , output logic [NUM_SUB-1:0][scale_exp_sum_bits-1:0] out_scale_exp_sum
    , output logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum
    , output logic [NUM_SUB-1:0][scale_mant_product_bits-1:0] out_scale_mant_product
);

    // E2M1 format constants
    localparam [1:0] FP4_EXP_BIAS = 2'd1;
    localparam FULL_MANT_W = 2;
    localparam UNSIGNED_PROD_W = 4;
    localparam SIGNED_PROD_W = 5;

    // Per-element decode signals (all 32 elements)
    logic [group_size-1:0] act_sign, wgt_sign;
    logic [group_size-1:0][1:0] act_exp_raw, wgt_exp_raw;
    logic [group_size-1:0] act_mant_raw, wgt_mant_raw;
    logic [group_size-1:0][1:0] act_exp, wgt_exp;
    logic [group_size-1:0][FULL_MANT_W-1:0] act_mant, wgt_mant;
    logic [group_size-1:0][UNSIGNED_PROD_W-1:0] prod_raw;
    logic signed [group_size-1:0][SIGNED_PROD_W-1:0] prod_sgn;
    logic [group_size-1:0][2:0] shamt;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_fixed;

    // Per-sub-block E4M3 scale decode
    logic [NUM_SUB-1:0][SCALE_EXP_BITS-1:0] scale_a_exp, scale_b_exp;
    logic [NUM_SUB-1:0][SCALE_MANT_BITS-1:0] scale_a_mant, scale_b_mant;
    logic [NUM_SUB-1:0][SCALE_MANT_BITS:0] scale_a_full_mant, scale_b_full_mant;

    logic [NUM_SUB-1:0][scale_exp_sum_bits-1:0] scale_exp_sum_pre;
    logic [NUM_SUB-1:0][scale_mant_product_bits-1:0] scale_mant_product_pre;
    logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum_pre;

    // Per-element E2M1 decode, multiply, fixed-point conversion (identical to NVFP4_MUL)
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            act_sign[i]     = nvfp4_a_elem[i][3];
            act_exp_raw[i]  = nvfp4_a_elem[i][2:1];
            act_mant_raw[i] = nvfp4_a_elem[i][0];

            wgt_sign[i]     = nvfp4_b_elem[i][3];
            wgt_exp_raw[i]  = nvfp4_b_elem[i][2:1];
            wgt_mant_raw[i] = nvfp4_b_elem[i][0];

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

        // Decode the two E4M3 scales per operand and form per-sub-block products
        for (int g = 0; g < NUM_SUB; g++) begin
            scale_a_exp[g]  = nvfp4_a_scale[g][6:3];
            scale_a_mant[g] = nvfp4_a_scale[g][2:0];
            scale_b_exp[g]  = nvfp4_b_scale[g][6:3];
            scale_b_mant[g] = nvfp4_b_scale[g][2:0];

            scale_a_full_mant[g] = {(scale_a_exp[g] != '0), scale_a_mant[g]};
            scale_b_full_mant[g] = {(scale_b_exp[g] != '0), scale_b_mant[g]};

            scale_mant_product_pre[g] = scale_a_full_mant[g] * scale_b_full_mant[g];
            scale_exp_sum_pre[g] = scale_exp_sum_bits'(scale_a_exp[g])
                                 + scale_exp_sum_bits'(scale_b_exp[g]);
        end
    end

    // One adder tree per sub-block (16 elements, per-stage 1-bit width growth)
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
