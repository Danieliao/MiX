module MiX45b_MXINT5_SystolicArray #(
    parameter group_size = 32
    , parameter wgt_group_size = 16
    , parameter sub_group_size = 8
    , parameter mxint_exp_bits = 8
    , parameter mxint_mant_bits = 5
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

    // MXINT4 weight inputs (left side): 2 groups per row
    , input logic [ROWS-1:0][group_size/wgt_group_size-1:0][mxint_exp_bits-1:0] mxint_exp_in
    , input logic signed [ROWS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_mant_in

    // INVMX activation inputs (top side): 1 group per column with 4 sub-group mantissas
    , input logic [COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_in
    , input logic [COLS-1:0][group_size-1:0] invmx_sign_in
    , input logic [COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_in
    , input logic [COLS-1:0][group_size/sub_group_size-1:0][invmx_mant_bits-1:0] invmx_mant_in

    // FP32 accumulator inputs (top side)
    , input logic [COLS-1:0] acc_sign_in
    , input logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_in
    , input logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_in

    // MXINT4 weight outputs (right side)
    , output logic [ROWS-1:0][group_size/wgt_group_size-1:0][mxint_exp_bits-1:0] mxint_exp_out
    , output logic signed [ROWS-1:0][group_size-1:0][mxint_mant_bits-1:0] mxint_mant_out

    // INVMX activation outputs (bottom side)
    , output logic [COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_out
    , output logic [COLS-1:0][group_size-1:0] invmx_sign_out
    , output logic [COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_out
    , output logic [COLS-1:0][group_size/sub_group_size-1:0][invmx_mant_bits-1:0] invmx_mant_out

    // FP32 accumulator outputs (bottom side)
    , output logic [COLS-1:0] acc_sign_out
    , output logic [COLS-1:0][fp_exp_bits-1:0] acc_exp_out
    , output logic [COLS-1:0][fp_mant_bits-1:0] acc_mant_out
);

    localparam NUM_WGT_GROUPS = group_size / wgt_group_size;
    localparam NUM_SUB_GROUPS = group_size / sub_group_size;

    // MXINT weight flows left-to-right across columns
    logic [ROWS-1:0][COLS:0][NUM_WGT_GROUPS-1:0][mxint_exp_bits-1:0] mxint_exp_wire;
    logic signed [ROWS-1:0][COLS:0][group_size-1:0][mxint_mant_bits-1:0] mxint_mant_wire;

    // INVMX activation flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0][max_base_exp_bits-1:0] invmx_max_base_exp_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0] invmx_sign_wire;
    logic [ROWS:0][COLS-1:0][group_size-1:0][invmx_exp_bits-1:0] invmx_exp_wire;
    logic [ROWS:0][COLS-1:0][NUM_SUB_GROUPS-1:0][invmx_mant_bits-1:0] invmx_mant_wire;

    // Accumulator flows top-to-bottom across rows
    logic [ROWS:0][COLS-1:0] acc_sign_wire;
    logic [ROWS:0][COLS-1:0][fp_exp_bits-1:0] acc_exp_wire;
    logic [ROWS:0][COLS-1:0][fp_mant_bits-1:0] acc_mant_wire;

    // Connect inputs to the wire mesh
    always_comb begin
        for (int r = 0; r < ROWS; r++) begin
            mxint_exp_wire[r][0]  = mxint_exp_in[r];
            mxint_mant_wire[r][0] = mxint_mant_in[r];
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
                MiX45b_MXINT5_PE #(
                    .group_size(group_size)
                    , .wgt_group_size(wgt_group_size)
                    , .sub_group_size(sub_group_size)
                    , .mxint_exp_bits(mxint_exp_bits)
                    , .mxint_mant_bits(mxint_mant_bits)
                    , .max_base_exp_bits(max_base_exp_bits)
                    , .invmx_exp_bits(invmx_exp_bits)
                    , .invmx_mant_bits(invmx_mant_bits)
                    , .fp_exp_bits(fp_exp_bits)
                    , .fp_mant_bits(fp_mant_bits)
                ) u_pe (
                    .clk(clk)
                    , .rst_n(rst_n)
                    , .acc_shift(acc_shift)
                    // MXINT weight from left
                    , .mxint_exp_in(mxint_exp_wire[r][c])
                    , .mxint_mant_in(mxint_mant_wire[r][c])
                    // INVMX activation from top
                    , .invmx_max_base_exp_in(invmx_max_base_exp_wire[r][c])
                    , .invmx_sign_in(invmx_sign_wire[r][c])
                    , .invmx_exp_in(invmx_exp_wire[r][c])
                    , .invmx_mant_in(invmx_mant_wire[r][c])
                    // Accumulator from top
                    , .acc_sign_in(acc_sign_wire[r][c])
                    , .acc_exp_in(acc_exp_wire[r][c])
                    , .acc_mant_in(acc_mant_wire[r][c])
                    // MXINT weight to right
                    , .mxint_exp_out(mxint_exp_wire[r][c+1])
                    , .mxint_mant_out(mxint_mant_wire[r][c+1])
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
            mxint_exp_out[r]  = mxint_exp_wire[r][COLS];
            mxint_mant_out[r] = mxint_mant_wire[r][COLS];
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
