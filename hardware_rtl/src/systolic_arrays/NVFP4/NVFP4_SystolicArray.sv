// Block-32 NVFP4 systolic array: 4x4 PEs, each PE a 32-element (two 16-elem
// sub-block) dot product → 512 MACs/cycle, matching the block-32 (4.25b) family
// (MXFP4, AMXFP4, MX+) for an iso-block-size comparison.
module NVFP4_SystolicArray #(
    parameter group_size = 32
    , parameter sub_group_size = 16
    , parameter scale_bits = 8
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

    , input logic [ROWS-1:0][NUM_SUB-1:0][scale_bits-1:0] nvfp4_a_scale_in
    , input logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] nvfp4_a_elem_in
    , input logic [COLS-1:0][NUM_SUB-1:0][scale_bits-1:0] nvfp4_b_scale_in
    , input logic [COLS-1:0][group_size-1:0][elem_bits-1:0] nvfp4_b_elem_in

    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    , output logic [ROWS-1:0][NUM_SUB-1:0][scale_bits-1:0] nvfp4_a_scale_out
    , output logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] nvfp4_a_elem_out
    , output logic [COLS-1:0][NUM_SUB-1:0][scale_bits-1:0] nvfp4_b_scale_out
    , output logic [COLS-1:0][group_size-1:0][elem_bits-1:0] nvfp4_b_elem_out

    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    // Activation flows left-to-right across columns
    logic [ROWS-1:0][COLS:0][NUM_SUB-1:0][scale_bits-1:0] a_scale_wire;
    logic [ROWS-1:0][COLS:0][group_size-1:0][elem_bits-1:0] a_elem_wire;

    // Weight flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0][NUM_SUB-1:0][scale_bits-1:0] b_scale_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0][elem_bits-1:0] b_elem_wire;

    // Accumulator flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            a_scale_wire[r][0] = nvfp4_a_scale_in[r];
            a_elem_wire[r][0]  = nvfp4_a_elem_in[r];
        end

        for (int c = 0; c < COLS; c++) begin
            b_scale_wire[0][c] = nvfp4_b_scale_in[c];
            b_elem_wire[0][c]  = nvfp4_b_elem_in[c];
            acc_sign_wire[0][c] = acc_sign_in[c];
            acc_exp_wire[0][c]  = acc_exp_in[c];
            acc_mant_wire[0][c] = acc_mant_in[c];
        end
    end

    // PE grid
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                NVFP4_PE #(
                    .group_size(group_size)
                    , .sub_group_size(sub_group_size)
                    , .scale_bits(scale_bits)
                    , .elem_bits(elem_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    , .nvfp4_a_scale_in(a_scale_wire[r][c])
                    , .nvfp4_a_elem_in(a_elem_wire[r][c])
                    , .nvfp4_b_scale_in(b_scale_wire[r][c])
                    , .nvfp4_b_elem_in(b_elem_wire[r][c])
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    , .nvfp4_a_scale_out(a_scale_wire[r][c+1])
                    , .nvfp4_a_elem_out(a_elem_wire[r][c+1])
                    , .nvfp4_b_scale_out(b_scale_wire[r+1][c])
                    , .nvfp4_b_elem_out(b_elem_wire[r+1][c])
                    , .acc_sign_out(acc_sign_wire[r+1][c])
                    , .acc_exp_out(acc_exp_wire[r+1][c])
                    , .acc_mant_out(acc_mant_wire[r+1][c])
                );
            end
        end
    endgenerate

    // Connect wire mesh to outputs
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            nvfp4_a_scale_out[r] = a_scale_wire[r][COLS];
            nvfp4_a_elem_out[r]  = a_elem_wire[r][COLS];
        end

        for (int c = 0; c < COLS; c++) begin
            nvfp4_b_scale_out[c] = b_scale_wire[ROWS][c];
            nvfp4_b_elem_out[c]  = b_elem_wire[ROWS][c];
            acc_sign_out[c]      = acc_sign_wire[ROWS][c];
            acc_exp_out[c]       = acc_exp_wire[ROWS][c];
            acc_mant_out[c]      = acc_mant_wire[ROWS][c];
        end
    end

endmodule
