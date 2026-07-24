module FP16_SystolicArray #(
    parameter fp16_bits = 16
    , parameter fp16_exp_bits = 5
    , parameter fp16_mant_bits = 10
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter ROWS = 32
    , parameter COLS = 32
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    // FP16 activation inputs (top side, one element per column)
    , input logic [COLS-1:0][fp16_bits-1:0] fp16_a_in

    // FP16 weight inputs (left side, one element per row)
    , input logic [ROWS-1:0][fp16_bits-1:0] fp16_b_in

    // FP32 accumulator inputs (top side)
    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    // FP16 activation outputs (bottom side)
    , output logic [COLS-1:0][fp16_bits-1:0] fp16_a_out

    // FP16 weight outputs (right side)
    , output logic [ROWS-1:0][fp16_bits-1:0] fp16_b_out

    // FP32 accumulator outputs (bottom side)
    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    // Activation flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0][fp16_bits-1:0] fp16_a_wire;

    // Weight flows left-to-right across columns
    logic [ROWS-1:0][COLS:0][fp16_bits-1:0] fp16_b_wire;

    // Accumulator flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int c = 0; c < COLS; c++) begin
            fp16_a_wire[0][c]  = fp16_a_in[c];
            acc_sign_wire[0][c] = acc_sign_in[c];
            acc_exp_wire[0][c]  = acc_exp_in[c];
            acc_mant_wire[0][c] = acc_mant_in[c];
        end

        for (int r = 0; r < ROWS; r++) begin
            fp16_b_wire[r][0] = fp16_b_in[r];
        end
    end

    // PE grid
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                FP16_PE #(
                    .fp16_bits(fp16_bits)
                    , .fp16_exp_bits(fp16_exp_bits)
                    , .fp16_mant_bits(fp16_mant_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    // Activation from top
                    , .fp16_a_in(fp16_a_wire[r][c])
                    // Weight from left
                    , .fp16_b_in(fp16_b_wire[r][c])
                    // Accumulator from top
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    // Activation to bottom
                    , .fp16_a_out(fp16_a_wire[r+1][c])
                    // Weight to right
                    , .fp16_b_out(fp16_b_wire[r][c+1])
                    // Accumulator to bottom
                    , .acc_sign_out(acc_sign_wire[r+1][c])
                    , .acc_exp_out(acc_exp_wire[r+1][c])
                    , .acc_mant_out(acc_mant_wire[r+1][c])
                );
            end
        end
    endgenerate

    // Connect wire mesh to outputs
    always_comb begin
        for (int c = 0; c < COLS; c++) begin
            fp16_a_out[c]  = fp16_a_wire[ROWS][c];
            acc_sign_out[c] = acc_sign_wire[ROWS][c];
            acc_exp_out[c]  = acc_exp_wire[ROWS][c];
            acc_mant_out[c] = acc_mant_wire[ROWS][c];
        end

        for (int r = 0; r < ROWS; r++) begin
            fp16_b_out[r] = fp16_b_wire[r][COLS];
        end
    end

endmodule
