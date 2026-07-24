module MiX425b_MXFP4_SystolicArray #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter max_base_exp_bits = 5
    , parameter invmx_exp_bits = 3
    , parameter invmx_mant_bits = 3
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter ROWS = 4
    , parameter COLS = 4
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    // MXFP4 weight inputs (left side, one group per row)
    , input logic [ROWS-1:0][shared_exp_bits-1:0] mxfp4_exp_in
    , input logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_elem_in

    // INVMX activation inputs (top side, one group per column)
    , input logic [COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_in
    , input logic [COLS-1:0][group_size-1:0] invmx_sign_in
    , input logic [COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_in
    , input logic [COLS-1:0][invmx_mant_bits-1:0] invmx_mant_in

    // FP32 accumulator inputs (top side)
    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    // MXFP4 weight outputs (right side)
    , output logic [ROWS-1:0][shared_exp_bits-1:0] mxfp4_exp_out
    , output logic [ROWS-1:0][group_size-1:0][elem_bits-1:0] mxfp4_elem_out

    // INVMX activation outputs (bottom side)
    , output logic [COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_out
    , output logic [COLS-1:0][group_size-1:0] invmx_sign_out
    , output logic [COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_out
    , output logic [COLS-1:0][invmx_mant_bits-1:0] invmx_mant_out

    // FP32 accumulator outputs (bottom side)
    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    // MXFP4 weight flows left-to-right across columns
    logic [ROWS-1:0][COLS:0][shared_exp_bits-1:0] mxfp4_exp_wire;
    logic [ROWS-1:0][COLS:0][group_size-1:0][elem_bits-1:0] mxfp4_elem_wire;

    // INVMX activation flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0] invmx_sign_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_wire;
    logic [ROWS:0][COLS-1:0][invmx_mant_bits-1:0] invmx_mant_wire;

    // Accumulator flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            mxfp4_exp_wire[r][0]  = mxfp4_exp_in[r];
            mxfp4_elem_wire[r][0] = mxfp4_elem_in[r];
        end

        for (int c = 0; c < COLS; c++) begin
            invmx_max_base_exp_wire[0][c] = invmx_max_base_exp_in[c];
            invmx_sign_wire[0][c]         = invmx_sign_in[c];
            invmx_exp_wire[0][c]          = invmx_exp_in[c];
            invmx_mant_wire[0][c]         = invmx_mant_in[c];
            acc_sign_wire[0][c]           = acc_sign_in[c];
            acc_exp_wire[0][c]            = acc_exp_in[c];
            acc_mant_wire[0][c]           = acc_mant_in[c];
        end
    end

    // PE grid
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                MiX425b_MXFP4_PE #(
                    .group_size(group_size)
                    , .shared_exp_bits(shared_exp_bits)
                    , .elem_bits(elem_bits)
                    , .max_base_exp_bits(max_base_exp_bits)
                    , .invmx_exp_bits(invmx_exp_bits)
                    , .invmx_mant_bits(invmx_mant_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    // MXFP4 weight from left
                    , .mxfp4_exp_in(mxfp4_exp_wire[r][c])
                    , .mxfp4_elem_in(mxfp4_elem_wire[r][c])
                    // INVMX activation from top
                    , .invmx_max_base_exp_in(invmx_max_base_exp_wire[r][c])
                    , .invmx_sign_in(invmx_sign_wire[r][c])
                    , .invmx_exp_in(invmx_exp_wire[r][c])
                    , .invmx_mant_in(invmx_mant_wire[r][c])
                    // Accumulator from top
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    // MXFP4 weight to right
                    , .mxfp4_exp_out(mxfp4_exp_wire[r][c+1])
                    , .mxfp4_elem_out(mxfp4_elem_wire[r][c+1])
                    // INVMX activation to bottom
                    , .invmx_max_base_exp_out(invmx_max_base_exp_wire[r+1][c])
                    , .invmx_sign_out(invmx_sign_wire[r+1][c])
                    , .invmx_exp_out(invmx_exp_wire[r+1][c])
                    , .invmx_mant_out(invmx_mant_wire[r+1][c])
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
        for (int r = 0; r < ROWS; r++) begin
            mxfp4_exp_out[r]  = mxfp4_exp_wire[r][COLS];
            mxfp4_elem_out[r] = mxfp4_elem_wire[r][COLS];
        end

        for (int c = 0; c < COLS; c++) begin
            invmx_max_base_exp_out[c] = invmx_max_base_exp_wire[ROWS][c];
            invmx_sign_out[c]         = invmx_sign_wire[ROWS][c];
            invmx_exp_out[c]          = invmx_exp_wire[ROWS][c];
            invmx_mant_out[c]         = invmx_mant_wire[ROWS][c];
            acc_sign_out[c]           = acc_sign_wire[ROWS][c];
            acc_exp_out[c]            = acc_exp_wire[ROWS][c];
            acc_mant_out[c]           = acc_mant_wire[ROWS][c];
        end
    end

endmodule
