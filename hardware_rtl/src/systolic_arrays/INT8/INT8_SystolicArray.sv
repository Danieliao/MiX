module INT8_SystolicArray #(
    parameter group_size = 32
    , parameter data_bits = 8
    , parameter acc_bits = 32
    , parameter ROWS = 4
    , parameter COLS = 4
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic signed [ROWS-1:0][group_size-1:0][data_bits-1:0] a_in
    , input logic signed [COLS-1:0][group_size-1:0][data_bits-1:0] b_in

    , input logic signed [COLS-1:0][acc_bits-1:0] acc_in

    , output logic signed [ROWS-1:0][group_size-1:0][data_bits-1:0] a_out
    , output logic signed [COLS-1:0][group_size-1:0][data_bits-1:0] b_out

    , output logic signed [COLS-1:0][acc_bits-1:0] acc_out
);

    // Activation flows left-to-right across columns
    logic signed [ROWS-1:0][COLS:0][group_size-1:0][data_bits-1:0] a_wire;

    // Weight flows top-to-bottom across rows
    logic signed [ROWS:0][COLS-1:0][group_size-1:0][data_bits-1:0] b_wire;

    // Accumulator flows top-to-bottom across rows
    logic signed [ROWS:0][COLS-1:0][acc_bits-1:0] acc_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            a_wire[r][0] = a_in[r];
        end

        for (int c = 0; c < COLS; c++) begin
            b_wire[0][c] = b_in[c];
            acc_wire[0][c] = acc_in[c];
        end
    end

    // PE grid
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                INT8_PE #(
                    .group_size(group_size)
                    , .data_bits(data_bits)
                    , .acc_bits(acc_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    , .a_in(a_wire[r][c])
                    , .b_in(b_wire[r][c])
                    , .acc_in(acc_wire[r][c])
                    , .a_out(a_wire[r][c+1])
                    , .b_out(b_wire[r+1][c])
                    , .acc_out(acc_wire[r+1][c])
                );
            end
        end
    endgenerate

    // Connect wire mesh to outputs
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            a_out[r] = a_wire[r][COLS];
        end

        for (int c = 0; c < COLS; c++) begin
            b_out[c] = b_wire[ROWS][c];
            acc_out[c] = acc_wire[ROWS][c];
        end
    end

endmodule
