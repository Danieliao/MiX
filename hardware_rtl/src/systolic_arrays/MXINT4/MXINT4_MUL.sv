module MXINT4_MUL #(
    parameter group_size = 32
    , parameter mxint_exp_bits = 8
    , parameter mxint_mant_bits = 4
    , parameter mult_out_bits = 2 * mxint_mant_bits
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)
    , parameter dot_out_bits = mult_out_bits + LOG2_GROUP_SIZE
) (
    input logic clk
    , input logic rst_n

    , input logic [mxint_exp_bits-1:0] mxint_a_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_a_mant
    , input logic [mxint_exp_bits-1:0] mxint_b_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_b_mant

    , output logic [mxint_exp_bits:0] out_exp
    , output logic signed [dot_out_bits-1:0] out_sum
);

    logic signed [group_size-1:0][mult_out_bits-1:0] mult_result;

    logic [mxint_exp_bits:0] combined_exp_pre;
    logic signed [dot_out_bits-1:0] out_sum_pre;

    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            mult_result[i] = mult_out_bits'(signed'(mxint_a_mant[i]))
                           * mult_out_bits'(signed'(mxint_b_mant[i]));
        end
    end

    always_comb begin
        combined_exp_pre = (mxint_exp_bits + 1)'(mxint_a_exp)
                         + (mxint_exp_bits + 1)'(mxint_b_exp);
    end

    // Adder tree with per-stage 1-bit width growth
    genvar s, i;
    generate
        for (s = 0; s <= LOG2_GROUP_SIZE; s++) begin : tree
            localparam int W = mult_out_bits + s;
            localparam int N = group_size >> s;
            logic signed [W-1:0] val [N-1:0];

            if (s == 0) begin : init
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = mult_result[i];
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