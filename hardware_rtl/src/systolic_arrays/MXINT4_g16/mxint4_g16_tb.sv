// Self-checking testbench for MXINT4_g16_PE.
//
// This is the block-16 MXINT4 format packed into a 32-element PE: the scale
// granularity stays 16 (so the format is still 4.5 b/element), and one PE
// consumes TWO 16-element sub-blocks, each carrying its own E8M0 exponent.
//
//     value = SUM_g [ (SUM_{i in g} a_mant[i]*b_mant[i]) * 2^(a_exp[g]+b_exp[g]-254) ]
//
// The PE normalizes each sub-block product and folds both into the running
// accumulator with one fused 3-input FP add. Output decode uses the same
// double-bias convention:  value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 254).
//
// A small relative tolerance is used because the fused 3-input add aligns the
// two sub-block results to the larger exponent and truncates.
module mxint4_g16_tb;
    localparam int group_size      = 32;
    localparam int sub_group_size  = 16;
    localparam int NUM_SUB         = group_size / sub_group_size;   // 2
    localparam int mxint_exp_bits  = 8;
    localparam int mxint_mant_bits = 4;
    localparam int fp_exp_bits     = 8;
    localparam int fp_mant_bits    = 23;
    localparam int NVEC            = 1000;
    localparam real EXP_OFFSET     = 254.0;   // 2 * E8M0 bias

    logic clk = 0, rst_n, acc_shift = 0;

    logic [NUM_SUB-1:0][mxint_exp_bits-1:0] a_exp, b_exp;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] a_mant, b_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [NUM_SUB-1:0][mxint_exp_bits-1:0] a_exp_o, b_exp_o;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] a_mant_o, b_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MXINT4_g16_PE #(
        .group_size(group_size), .sub_group_size(sub_group_size),
        .mxint_exp_bits(mxint_exp_bits), .mxint_mant_bits(mxint_mant_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .mxint_a_exp_in(a_exp), .mxint_a_mant_in(a_mant),
        .mxint_b_exp_in(b_exp), .mxint_b_mant_in(b_mant),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .mxint_a_exp_out(a_exp_o), .mxint_a_mant_out(a_mant_o),
        .mxint_b_exp_out(b_exp_o), .mxint_b_mant_out(b_mant_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    function automatic logic signed [mxint_mant_bits-1:0] rand_mant();
        return mxint_mant_bits'(($urandom() % 15) - 7);
    endfunction

    int seed = 32'h9C10B16;

    initial begin
        int fails;
        longint dot_g;
        real expval, dutval, tol, absval;
        logic signed [mxint_mant_bits-1:0] av, bv;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            expval = 0.0;
            absval = 0.0;

            for (int g = 0; g < NUM_SUB; g++) begin
                // per-sub-block exponents; sum kept <= ~234 so the 8-bit
                // accumulator exponent field cannot saturate (see mxint4_tb)
                a_exp[g] = mxint_exp_bits'(100 + ($urandom() % 18));
                b_exp[g] = mxint_exp_bits'(100 + ($urandom() % 18));

                dot_g = 0;
                for (int i = 0; i < sub_group_size; i++) begin
                    av = rand_mant();  bv = rand_mant();
                    a_mant[g*sub_group_size + i] = av;
                    b_mant[g*sub_group_size + i] = bv;
                    dot_g += longint'(av) * longint'(bv);
                end

                expval += real'(dot_g)
                        * $pow(2.0, real'(a_exp[g]) + real'(b_exp[g]) - EXP_OFFSET);
                absval += ((dot_g < 0) ? real'(-dot_g) : real'(dot_g))
                        * $pow(2.0, real'(a_exp[g]) + real'(b_exp[g]) - EXP_OFFSET);
            end

            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;

            if (acc_exp_o == 0 && acc_mant_o == 0)
                dutval = 0.0;
            else
                dutval = (acc_sign_o ? -1.0 : 1.0)
                       * (1.0 + real'(acc_mant_o) / $pow(2.0, real'(fp_mant_bits)))
                       * $pow(2.0, real'(acc_exp_o) - EXP_OFFSET);

            // tolerance relative to the magnitude actually summed (catches
            // cancellation cases where expval is near zero)
            tol = 1e-6 * absval + 1e-30;
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
