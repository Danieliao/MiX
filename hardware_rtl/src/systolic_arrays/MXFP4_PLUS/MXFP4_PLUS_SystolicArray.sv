module MXFP4_PLUS_SystolicArray #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter bm_idx_bits = $clog2(group_size)
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter ROWS = 4
    , parameter COLS = 4
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [ROWS-1:0][shared_exp_bits-1:0] a_exp_in
    , input logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] a_elem_in
    , input logic [ROWS-1:0][bm_idx_bits-1:0] a_bm_idx_in
    , input logic [COLS-1:0][shared_exp_bits-1:0] b_exp_in
    , input logic [COLS-1:0][group_size-1:0][elem_bits-1:0] b_elem_in
    , input logic [COLS-1:0][bm_idx_bits-1:0] b_bm_idx_in

    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    , output logic [ROWS-1:0][shared_exp_bits-1:0] a_exp_out
    , output logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] a_elem_out
    , output logic [ROWS-1:0][bm_idx_bits-1:0] a_bm_idx_out
    , output logic [COLS-1:0][shared_exp_bits-1:0] b_exp_out
    , output logic [COLS-1:0][group_size-1:0][elem_bits-1:0] b_elem_out
    , output logic [COLS-1:0][bm_idx_bits-1:0] b_bm_idx_out

    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    // Activation flows left-to-right across columns
    logic [ROWS-1:0][COLS:0][shared_exp_bits-1:0] a_exp_wire;
    logic [ROWS-1:0][COLS:0][group_size-1:0][elem_bits-1:0] a_elem_wire;
    logic [ROWS-1:0][COLS:0][bm_idx_bits-1:0] a_bm_idx_wire;

    // Weight flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0][shared_exp_bits-1:0] b_exp_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0][elem_bits-1:0] b_elem_wire;
    logic [ROWS:0][COLS-1:0][bm_idx_bits-1:0] b_bm_idx_wire;

    // Accumulator flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            a_exp_wire[r][0]    = a_exp_in[r];
            a_elem_wire[r][0]   = a_elem_in[r];
            a_bm_idx_wire[r][0] = a_bm_idx_in[r];
        end

        for (int c = 0; c < COLS; c++) begin
            b_exp_wire[0][c]    = b_exp_in[c];
            b_elem_wire[0][c]   = b_elem_in[c];
            b_bm_idx_wire[0][c] = b_bm_idx_in[c];
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
                MXFP4_PLUS_PE #(
                    .group_size(group_size)
                    , .shared_exp_bits(shared_exp_bits)
                    , .elem_bits(elem_bits)
                    , .bm_idx_bits(bm_idx_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    , .a_exp_in(a_exp_wire[r][c])
                    , .a_elem_in(a_elem_wire[r][c])
                    , .a_bm_idx_in(a_bm_idx_wire[r][c])
                    , .b_exp_in(b_exp_wire[r][c])
                    , .b_elem_in(b_elem_wire[r][c])
                    , .b_bm_idx_in(b_bm_idx_wire[r][c])
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    , .a_exp_out(a_exp_wire[r][c+1])
                    , .a_elem_out(a_elem_wire[r][c+1])
                    , .a_bm_idx_out(a_bm_idx_wire[r][c+1])
                    , .b_exp_out(b_exp_wire[r+1][c])
                    , .b_elem_out(b_elem_wire[r+1][c])
                    , .b_bm_idx_out(b_bm_idx_wire[r+1][c])
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
            a_exp_out[r]    = a_exp_wire[r][COLS];
            a_elem_out[r]   = a_elem_wire[r][COLS];
            a_bm_idx_out[r] = a_bm_idx_wire[r][COLS];
        end

        for (int c = 0; c < COLS; c++) begin
            b_exp_out[c]    = b_exp_wire[ROWS][c];
            b_elem_out[c]   = b_elem_wire[ROWS][c];
            b_bm_idx_out[c] = b_bm_idx_wire[ROWS][c];
            acc_sign_out[c] = acc_sign_wire[ROWS][c];
            acc_exp_out[c]  = acc_exp_wire[ROWS][c];
            acc_mant_out[c] = acc_mant_wire[ROWS][c];
        end
    end

endmodule
