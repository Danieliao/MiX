module INT8_MUL #(
    parameter group_size = 32
    , parameter data_bits = 8
    , parameter mult_out_bits = 2 * data_bits
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)
    , parameter dot_out_bits = mult_out_bits + LOG2_GROUP_SIZE
) (
    input logic clk
    , input logic rst_n

    , input logic signed [group_size-1:0][data_bits-1:0] a_in
    , input logic signed [group_size-1:0][data_bits-1:0] b_in

    , output logic signed [dot_out_bits-1:0] out_sum
);

    logic signed [group_size-1:0][mult_out_bits-1:0] mult_result;

    logic signed [dot_out_bits-1:0] out_sum_pre;

    // Parallel multiplies
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            mult_result[i] = mult_out_bits'(signed'(a_in[i]))
                           * mult_out_bits'(signed'(b_in[i]));
        end
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

    // Register output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_sum <= '0;
        end else begin
            out_sum <= out_sum_pre;
        end
    end

endmodule
