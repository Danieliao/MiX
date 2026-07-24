// Self-checking testbench for MiX45b_MXINT5_PE
// (MiX-4.5b INVMX activation x MXINT5 weight, block_size = 32).
//
// Structure: the 32-element block holds
//   * 2 weight groups of 16 elements, each with its own E8M0 exponent, and
//   * 4 INVMX sub-groups of 8 elements, each with its own shared 3-bit mantissa.
//
// Datapath (replayed exactly by the reference model below):
//   signed_mant[i] = invmx_exp[i]==7 ? 0 : (+/-) w_mant[i]
//   shifted[i]     = (signed_mant[i] << 3) >>> invmx_exp[i]      (truncating)
//   sub_acc[g]     = SUM of the 8 shifted values in sub-group g
//   sub_mult[g]    = sub_acc[g] * {1,mmm}[g]                     (8..15)
//   wgt_sum[0]     = sub_mult[0] + sub_mult[1]   (elements  0-15)
//   wgt_sum[1]     = sub_mult[2] + sub_mult[3]   (elements 16-31)
//   out_exp[w]     = mxint_exp[w] + max_base_exp - 6
//
//   value = SUM_w wgt_sum[w] * 2^(out_exp[w] - 127)
//
// Only the weight side carries the E8M0 bias, so the accumulator decodes with a
// SINGLE bias: value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 127).
// The PE normalizes both weight-group results and folds them into the running
// accumulator with one fused 3-input FP add, hence the small tolerance.
module mix45b_mxint5_tb;
    localparam int group_size        = 32;
    localparam int wgt_group_size    = 16;
    localparam int sub_group_size    = 8;
    localparam int NUM_WGT           = group_size / wgt_group_size;   // 2
    localparam int NUM_SUB           = group_size / sub_group_size;   // 4
    localparam int mxint_exp_bits    = 8;
    localparam int mxint_mant_bits   = 5;
    localparam int max_base_exp_bits = 5;
    localparam int invmx_exp_bits    = 3;
    localparam int invmx_mant_bits   = 3;
    localparam int fp_exp_bits       = 8;
    localparam int fp_mant_bits      = 23;
    localparam int FRAC_BITS         = 3;    // hardcoded in the RTL
    localparam int TOTAL_FIXED_SCALE = FRAC_BITS + invmx_mant_bits;   // 6
    localparam int MANT_LIM          = 15;   // |w_mant| <= 15 for 5-bit
    localparam int NVEC              = 1000;
    localparam real EXP_OFFSET       = 127.0;

    logic clk = 0, rst_n, acc_shift = 0;

    logic [NUM_WGT-1:0][mxint_exp_bits-1:0] mxint_exp;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant;
    logic [max_base_exp_bits-1:0] max_base_exp;
    logic [group_size-1:0] invmx_sign;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp;
    logic [NUM_SUB-1:0][invmx_mant_bits-1:0] invmx_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [NUM_WGT-1:0][mxint_exp_bits-1:0] mxint_exp_o;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant_o;
    logic [max_base_exp_bits-1:0] max_base_exp_o;
    logic [group_size-1:0] invmx_sign_o;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_o;
    logic [NUM_SUB-1:0][invmx_mant_bits-1:0] invmx_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MiX45b_MXINT5_PE #(
        .group_size(group_size), .wgt_group_size(wgt_group_size),
        .sub_group_size(sub_group_size), .mxint_exp_bits(mxint_exp_bits),
        .mxint_mant_bits(mxint_mant_bits), .max_base_exp_bits(max_base_exp_bits),
        .invmx_exp_bits(invmx_exp_bits), .invmx_mant_bits(invmx_mant_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .mxint_exp_in(mxint_exp), .mxint_mant_in(mxint_mant),
        .invmx_max_base_exp_in(max_base_exp), .invmx_sign_in(invmx_sign),
        .invmx_exp_in(invmx_exp), .invmx_mant_in(invmx_mant),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .mxint_exp_out(mxint_exp_o), .mxint_mant_out(mxint_mant_o),
        .invmx_max_base_exp_out(max_base_exp_o), .invmx_sign_out(invmx_sign_o),
        .invmx_exp_out(invmx_exp_o), .invmx_mant_out(invmx_mant_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    int seed = 32'h45B15D8;

    initial begin
        int fails;
        int sm, sh;
        int sub_acc [NUM_SUB];
        longint sub_mult [NUM_SUB];
        longint wgt_sum;
        int cexp;
        real expval, dutval, absval, tol, sc;
        logic signed [mxint_mant_bits-1:0] wm;
        logic [invmx_exp_bits-1:0] ie;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            max_base_exp = max_base_exp_bits'($urandom() % 32);
            for (int w = 0; w < NUM_WGT; w++)
                mxint_exp[w] = mxint_exp_bits'(100 + ($urandom() % 41));   // [100,140]
            for (int g = 0; g < NUM_SUB; g++)
                invmx_mant[g] = invmx_mant_bits'($urandom() % 8);

            // stimulus + per-sub-group integer reference
            for (int g = 0; g < NUM_SUB; g++) begin
                sub_acc[g] = 0;
                for (int j = 0; j < sub_group_size; j++) begin
                    automatic int i = g*sub_group_size + j;
                    wm = mxint_mant_bits'(($urandom() % (2*MANT_LIM+1)) - MANT_LIM);
                    ie = invmx_exp_bits'($urandom() % 8);   // 7 = zero element
                    mxint_mant[i] = wm;
                    invmx_exp[i]  = ie;
                    invmx_sign[i] = $urandom() % 2;

                    if (ie == 3'b111)       sm = 0;
                    else if (invmx_sign[i]) sm = -int'(wm);
                    else                    sm =  int'(wm);
                    sh = (sm <<< FRAC_BITS) >>> ie;
                    sub_acc[g] += sh;
                end
                sub_mult[g] = longint'(sub_acc[g]) * longint'(8 + invmx_mant[g]);
            end

            // combine per weight group and scale
            expval = 0.0;
            absval = 0.0;
            for (int w = 0; w < NUM_WGT; w++) begin
                wgt_sum = sub_mult[2*w] + sub_mult[2*w + 1];
                cexp    = int'(mxint_exp[w]) + int'(max_base_exp) - TOTAL_FIXED_SCALE;
                sc      = $pow(2.0, real'(cexp) - EXP_OFFSET);
                expval += real'(wgt_sum) * sc;
                absval += ((wgt_sum < 0) ? real'(-wgt_sum) : real'(wgt_sum)) * sc;
            end

            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;

            if (acc_exp_o == 0 && acc_mant_o == 0)
                dutval = 0.0;
            else
                dutval = (acc_sign_o ? -1.0 : 1.0)
                       * (1.0 + real'(acc_mant_o) / $pow(2.0, real'(fp_mant_bits)))
                       * $pow(2.0, real'(acc_exp_o) - EXP_OFFSET);

            tol = 1e-6 * absval + 1e-40;
            if (((dutval - expval) > tol) || ((expval - dutval) > tol)) begin
                fails++;
                if (fails <= 8)
                    $display("  FAIL[%0d] exp=%0.10g dut=%0.10g tol=%0.3g", n, expval, dutval, tol);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors within tolerance", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
