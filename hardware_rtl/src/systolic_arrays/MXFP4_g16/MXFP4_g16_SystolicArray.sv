// Block-32-PE MXFP4 (4.5b) systolic array: 4x4 PEs, 32 elements/PE (two 16-elem
// E8M0 sub-blocks) → 512 MACs/cycle, iso-PE-size with the block-32 family while
// keeping the per-16 E8M0 scale (still 4.5b).
module MXFP4_g16_SystolicArray #(
    parameter group_size = 32
    , parameter sub_group_size = 16
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter ROWS = 4
    , parameter COLS = 4
    , parameter NUM_SUB = group_size / sub_group_size
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [ROWS-1:0][NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_a_exp_in
    , input logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_a_elem_in
    , input logic [COLS-1:0][NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_b_exp_in
    , input logic [COLS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_b_elem_in

    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    , output logic [ROWS-1:0][NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_a_exp_out
    , output logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_a_elem_out
    , output logic [COLS-1:0][NUM_SUB-1:0][shared_exp_bits-1:0] mxfp4_b_exp_out
    , output logic [COLS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_b_elem_out

    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    logic [ROWS-1:0][COLS:0][NUM_SUB-1:0][shared_exp_bits-1:0] a_exp_wire;
    logic [ROWS-1:0][COLS:0][group_size-1:0][elem_bits-1:0] a_elem_wire;
    logic [ROWS:0][COLS-1:0][NUM_SUB-1:0][shared_exp_bits-1:0] b_exp_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0][elem_bits-1:0] b_elem_wire;
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            a_exp_wire[r][0]  = mxfp4_a_exp_in[r];
            a_elem_wire[r][0] = mxfp4_a_elem_in[r];
        end
        for (int c = 0; c < COLS; c++) begin
            b_exp_wire[0][c]  = mxfp4_b_exp_in[c];
            b_elem_wire[0][c] = mxfp4_b_elem_in[c];
            acc_sign_wire[0][c] = acc_sign_in[c];
            acc_exp_wire[0][c]  = acc_exp_in[c];
            acc_mant_wire[0][c] = acc_mant_in[c];
        end
    end

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                MXFP4_g16_PE #(
                    .group_size(group_size)
                    , .sub_group_size(sub_group_size)
                    , .shared_exp_bits(shared_exp_bits)
                    , .elem_bits(elem_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    , .mxfp4_a_exp_in(a_exp_wire[r][c])
                    , .mxfp4_a_elem_in(a_elem_wire[r][c])
                    , .mxfp4_b_exp_in(b_exp_wire[r][c])
                    , .mxfp4_b_elem_in(b_elem_wire[r][c])
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    , .mxfp4_a_exp_out(a_exp_wire[r][c+1])
                    , .mxfp4_a_elem_out(a_elem_wire[r][c+1])
                    , .mxfp4_b_exp_out(b_exp_wire[r+1][c])
                    , .mxfp4_b_elem_out(b_elem_wire[r+1][c])
                    , .acc_sign_out(acc_sign_wire[r+1][c])
                    , .acc_exp_out(acc_exp_wire[r+1][c])
                    , .acc_mant_out(acc_mant_wire[r+1][c])
                );
            end
        end
    endgenerate

    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            mxfp4_a_exp_out[r]  = a_exp_wire[r][COLS];
            mxfp4_a_elem_out[r] = a_elem_wire[r][COLS];
        end
        for (int c = 0; c < COLS; c++) begin
            mxfp4_b_exp_out[c]  = b_exp_wire[ROWS][c];
            mxfp4_b_elem_out[c] = b_elem_wire[ROWS][c];
            acc_sign_out[c]     = acc_sign_wire[ROWS][c];
            acc_exp_out[c]      = acc_exp_wire[ROWS][c];
            acc_mant_out[c]     = acc_mant_wire[ROWS][c];
        end
    end

endmodule
