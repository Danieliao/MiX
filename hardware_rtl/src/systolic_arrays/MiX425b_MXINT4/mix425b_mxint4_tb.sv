// Self-checking testbench for MiX425b_MXINT4_PE
// (MiX-4.25b INVMX activation x MXINT4 weight, block_size = 32).
//
// INVMX shares ONE mantissa per activation group while each element carries its
// own 3-bit exponent offset, so the shared mantissa factors out of the dot
// product and the per-element work becomes a shift-and-add:
//
//   dot = shared_mant * 2^(mxint_exp + max_base_exp) * SUM_i (+/- w_mant[i] >> invmx_exp[i])
//
// invmx_exp == 3'b111 encodes a zero element. The MUL appends frac_bits(=3)
// fractional zeros before the arithmetic right shift, sums the 32 terms, then
// multiplies once by the shared mantissa {1,mmm} (4-bit, 8..15), and subtracts
// TOTAL_FIXED_SCALE = 3 + 3 = 6 from the combined exponent.
//
// Only the WEIGHT side carries an E8M0 bias (127); max_base_exp is unbiased, so
// the accumulator decodes with a SINGLE bias:
//   value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 127)
//
// The reference model below replays the exact same integer operations (including
// the truncating arithmetic shift), so the comparison is EXACT.
module mix425b_mxint4_tb;
    localparam int group_size        = 32;
    localparam int mxint_exp_bits    = 8;
    localparam int mxint_mant_bits   = 4;
    localparam int max_base_exp_bits = 5;
    localparam int invmx_exp_bits    = 3;
    localparam int invmx_mant_bits   = 3;
    localparam int fp_exp_bits       = 8;
    localparam int fp_mant_bits      = 23;
    localparam int FRAC_BITS         = 3;   // hardcoded in the RTL
    localparam int TOTAL_FIXED_SCALE = FRAC_BITS + invmx_mant_bits;  // 6
    localparam int NVEC              = 1000;
    localparam real EXP_OFFSET       = 127.0;  // single E8M0 bias (weight side only)

    logic clk = 0, rst_n, acc_shift = 0;

    logic [mxint_exp_bits-1:0] mxint_exp;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant;
    logic [max_base_exp_bits-1:0] max_base_exp;
    logic [group_size-1:0] invmx_sign;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp;
    logic [invmx_mant_bits-1:0] invmx_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [mxint_exp_bits-1:0] mxint_exp_o;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant_o;
    logic [max_base_exp_bits-1:0] max_base_exp_o;
    logic [group_size-1:0] invmx_sign_o;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_o;
    logic [invmx_mant_bits-1:0] invmx_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MiX425b_MXINT4_PE #(
        .group_size(group_size), .mxint_exp_bits(mxint_exp_bits),
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

    int seed = 32'h4251B04;

    initial begin
        int fails;
        int sm, sh, acc_int;
        longint out_sum_ref;
        int combined_exp;
        real expval, dutval, tol;
        logic signed [mxint_mant_bits-1:0] wm;
        logic [invmx_exp_bits-1:0] ie;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            // combined_exp = mxint_exp + max_base_exp - 6 stays well inside [0,255]
            mxint_exp    = mxint_exp_bits'(100 + ($urandom() % 41));    // [100,140]
            max_base_exp = max_base_exp_bits'($urandom() % 32);         // [0,31]
            invmx_mant   = invmx_mant_bits'($urandom() % 8);

            acc_int = 0;
            for (int i = 0; i < group_size; i++) begin
                // weight mantissa restricted to [-7,7]: the RTL negates in
                // mxint_mant_bits width, so -8 would wrap
                wm = mxint_mant_bits'(($urandom() % 15) - 7);
                ie = invmx_exp_bits'($urandom() % 8);   // 7 encodes a zero element
                mxint_mant[i] = wm;
                invmx_exp[i]  = ie;
                invmx_sign[i] = $urandom() % 2;

                // --- reference: identical integer ops to the RTL ---
                if (ie == 3'b111)      sm = 0;
                else if (invmx_sign[i]) sm = -int'(wm);
                else                    sm =  int'(wm);
                sh = (sm <<< FRAC_BITS) >>> ie;   // arithmetic (truncating) shift
                acc_int += sh;
            end

            out_sum_ref  = longint'(acc_int) * longint'(8 + invmx_mant);
            combined_exp = int'(mxint_exp) + int'(max_base_exp) - TOTAL_FIXED_SCALE;
            expval       = real'(out_sum_ref)
                         * $pow(2.0, real'(combined_exp) - EXP_OFFSET);

            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;

            if (acc_exp_o == 0 && acc_mant_o == 0)
                dutval = 0.0;
            else
                dutval = (acc_sign_o ? -1.0 : 1.0)
                       * (1.0 + real'(acc_mant_o) / $pow(2.0, real'(fp_mant_bits)))
                       * $pow(2.0, real'(acc_exp_o) - EXP_OFFSET);

            tol = 1e-9 * ((expval < 0.0) ? -expval : expval) + 1e-40;
            if (((dutval - expval) > tol) || ((expval - dutval) > tol)) begin
                fails++;
                if (fails <= 8)
                    $display("  FAIL[%0d] out_sum=%0d cexp=%0d exp=%0.10g dut=%0.10g",
                             n, out_sum_ref, combined_exp, expval, dutval);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors EXACT match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
