module MXINT4_SystolicArray #(
    parameter group_size = 32
    , parameter mxint_exp_bits = 8
    , parameter mxint_mant_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter ROWS = 4
    , parameter COLS = 4
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [ROWS-1:0][mxint_exp_bits-1:0] mxint_a_exp_in
    , input logic signed [ROWS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_a_mant_in
    , input logic [COLS-1:0][mxint_exp_bits-1:0] mxint_b_exp_in
    , input logic signed [COLS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_b_mant_in

    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    , output logic [ROWS-1:0][mxint_exp_bits-1:0] mxint_a_exp_out
    , output logic signed [ROWS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_a_mant_out
    , output logic [COLS-1:0][mxint_exp_bits-1:0] mxint_b_exp_out
    , output logic signed [COLS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_b_mant_out

    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    logic [ROWS-1:0][COLS:0][mxint_exp_bits-1:0] mxint_a_exp_wire;
    logic signed [ROWS-1:0][COLS:0][group_size-1:0][mxint_mant_bits-1:0] mxint_a_mant_wire;

    logic [ROWS:0][COLS-1:0][mxint_exp_bits-1:0] mxint_b_exp_wire;
    logic signed [ROWS:0][COLS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_b_mant_wire;

    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            mxint_a_exp_wire[r][0] = mxint_a_exp_in[r];
            mxint_a_mant_wire[r][0] = mxint_a_mant_in[r];
        end

        for (int c = 0; c < COLS; c++) begin
            mxint_b_exp_wire[0][c] = mxint_b_exp_in[c];
            mxint_b_mant_wire[0][c] = mxint_b_mant_in[c];
            acc_sign_wire[0][c] = acc_sign_in[c];
            acc_exp_wire[0][c] = acc_exp_in[c];
            acc_mant_wire[0][c] = acc_mant_in[c];
        end
    end

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                MXINT4_PE #(
                    .group_size(group_size)
                    , .mxint_exp_bits(mxint_exp_bits)
                    , .mxint_mant_bits(mxint_mant_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    , .mxint_a_exp_in(mxint_a_exp_wire[r][c])
                    , .mxint_a_mant_in(mxint_a_mant_wire[r][c])
                    , .mxint_b_exp_in(mxint_b_exp_wire[r][c])
                    , .mxint_b_mant_in(mxint_b_mant_wire[r][c])
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    , .mxint_a_exp_out(mxint_a_exp_wire[r][c+1])
                    , .mxint_a_mant_out(mxint_a_mant_wire[r][c+1])
                    , .mxint_b_exp_out(mxint_b_exp_wire[r+1][c])
                    , .mxint_b_mant_out(mxint_b_mant_wire[r+1][c])
                    , .acc_sign_out(acc_sign_wire[r+1][c])
                    , .acc_exp_out(acc_exp_wire[r+1][c])
                    , .acc_mant_out(acc_mant_wire[r+1][c])
                );
            end
        end
    endgenerate

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            mxint_a_exp_out[r] = mxint_a_exp_wire[r][COLS];
            mxint_a_mant_out[r] = mxint_a_mant_wire[r][COLS];
        end

        for (int c = 0; c < COLS; c++) begin
            mxint_b_exp_out[c] = mxint_b_exp_wire[ROWS][c];
            mxint_b_mant_out[c] = mxint_b_mant_wire[ROWS][c];
            acc_sign_out[c] = acc_sign_wire[ROWS][c];
            acc_exp_out[c] = acc_exp_wire[ROWS][c];
            acc_mant_out[c] = acc_mant_wire[ROWS][c];
        end
    end

endmodule