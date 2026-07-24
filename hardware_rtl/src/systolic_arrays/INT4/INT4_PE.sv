module INT4_PE #(
    parameter group_size = 32
    , parameter data_bits = 4
    , parameter acc_bits = 32
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic signed [group_size-1:0][data_bits-1:0] a_in
    , input logic signed [group_size-1:0][data_bits-1:0] b_in

    , input logic signed [acc_bits-1:0] acc_in

    , output logic signed [group_size-1:0][data_bits-1:0] a_out
    , output logic signed [group_size-1:0][data_bits-1:0] b_out

    , output logic signed [acc_bits-1:0] acc_out
);

    localparam DOT_OUT_BITS = (2 * data_bits) + $clog2(group_size);

    // MUL output
    logic signed [DOT_OUT_BITS-1:0] mul_out_sum;

    // Local INT32 accumulator
    logic signed [acc_bits-1:0] local_acc;

    // ----------------------------------------------------------------
    // Stage 1: Parallel multiply + adder tree (registered in MUL)
    // ----------------------------------------------------------------
    INT4_MUL #(
        .group_size(group_size)
        , .data_bits(data_bits)
    ) u_int4_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .a_in(a_in)
        , .b_in(b_in)
        , .out_sum(mul_out_sum)
    );

    // ----------------------------------------------------------------
    // Stage 2: INT32 accumulation
    // ----------------------------------------------------------------
    assign acc_out = local_acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_acc <= '0;
        end else if (acc_shift) begin
            local_acc <= acc_in;
        end else begin
            local_acc <= local_acc + acc_bits'(signed'(mul_out_sum));
        end
    end

    // ----------------------------------------------------------------
    // Systolic pass-through: activation flows right, weight flows down
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
        end else begin
            a_out <= a_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_out <= '0;
        end else begin
            b_out <= b_in;
        end
    end

endmodule
