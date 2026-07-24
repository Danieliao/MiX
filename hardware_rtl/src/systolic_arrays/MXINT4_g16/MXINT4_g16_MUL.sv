// Block-32-PE MXINT4 (4.5b): two 16-element MXINT4 sub-blocks per PE, each KEEPING
// its own E8M0 scale (scale block stays 16 -> still 4.5b; PE width is 32). Emits
// one (exp, sum) per sub-block.
module MXINT4_g16_MUL #(
    parameter group_size = 32
    , parameter sub_group_size = 16
    , parameter mxint_exp_bits = 8
    , parameter mxint_mant_bits = 4
    , parameter mult_out_bits = 2 * mxint_mant_bits             // 8
    , parameter NUM_SUB = group_size / sub_group_size           // 2
    , parameter LOG2_SUB_SIZE = $clog2(sub_group_size)          // 4
    , parameter dot_out_bits = mult_out_bits + LOG2_SUB_SIZE    // 12
) (
    input logic clk
    , input logic rst_n

    , input logic [NUM_SUB-1:0][mxint_exp_bits-1:0] mxint_a_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_a_mant
    , input logic [NUM_SUB-1:0][mxint_exp_bits-1:0] mxint_b_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_b_mant

    , output logic [NUM_SUB-1:0][mxint_exp_bits:0] out_exp
    , output logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum
);

    logic signed [group_size-1:0][mult_out_bits-1:0] mult_result;
    logic [NUM_SUB-1:0][mxint_exp_bits:0] combined_exp_pre;
    logic signed [NUM_SUB-1:0][dot_out_bits-1:0] out_sum_pre;

    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            mult_result[i] = mult_out_bits'(signed'(mxint_a_mant[i]))
                           * mult_out_bits'(signed'(mxint_b_mant[i]));
        end
        for (int g = 0; g < NUM_SUB; g++) begin
            combined_exp_pre[g] = (mxint_exp_bits + 1)'(mxint_a_exp[g])
                                + (mxint_exp_bits + 1)'(mxint_b_exp[g]);
        end
    end

    // One adder tree per 16-element sub-block
    genvar g, s, i;
    generate
        for (g = 0; g < NUM_SUB; g++) begin : sub
            for (s = 0; s <= LOG2_SUB_SIZE; s++) begin : tree
                localparam int W = mult_out_bits + s;
                localparam int N = sub_group_size >> s;
                logic signed [W-1:0] val [N-1:0];

                if (s == 0) begin : init
                    for (i = 0; i < N; i++) begin : elem
                        assign val[i] = mult_result[g * sub_group_size + i];
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
