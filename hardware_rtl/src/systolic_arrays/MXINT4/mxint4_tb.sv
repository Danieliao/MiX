// Self-checking testbench for MXINT4_PE (OCP MX INT4, block_size = 32).
//
// Format: one E8M0 shared exponent per 32-element block plus per-element 4-bit
// two's-complement mantissas. The block dot product is therefore
//     value = (SUM_i a_mant[i]*b_mant[i]) * 2^(a_exp-127) * 2^(b_exp-127)
// The MUL emits out_exp = a_exp + b_exp and the integer mantissa dot product;
// the PE normalizes it into the custom FP32 accumulator, whose exponent field
// carries the DOUBLE bias (2*127 = 254). Hence the output decode:
//     value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 254)
//
// The mantissa dot product is at most 13 bits wide, so it fits exactly in the
// 24-bit FP32 significand -- the comparison is exact up to a tiny epsilon.
module mxint4_tb;
    localparam int group_size      = 32;
    localparam int mxint_exp_bits  = 8;
    localparam int mxint_mant_bits = 4;
    localparam int fp_exp_bits     = 8;
    localparam int fp_mant_bits    = 23;
    localparam int NVEC            = 1000;
    localparam real EXP_OFFSET     = 254.0;   // 2 * E8M0 bias

    logic clk = 0, rst_n, acc_shift = 0;

    logic [mxint_exp_bits-1:0] a_exp, b_exp;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] a_mant, b_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [mxint_exp_bits-1:0] a_exp_o, b_exp_o;
    logic signed [group_size-1:0][mxint_mant_bits-1:0] a_mant_o, b_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MXINT4_PE #(
        .group_size(group_size), .mxint_exp_bits(mxint_exp_bits),
        .mxint_mant_bits(mxint_mant_bits),
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

    // 4-bit two's-complement mantissa in [-7, 7]
    function automatic logic signed [mxint_mant_bits-1:0] rand_mant();
        return mxint_mant_bits'(($urandom() % 15) - 7);
    endfunction

    int seed = 32'h51EED04;

    initial begin
        int fails;
        longint dot;
        real expval, dutval, tol, scale;
        logic signed [mxint_mant_bits-1:0] av, bv;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            // The accumulator exponent field is 8 bits and holds the DOUBLE-biased
            // value a_exp + b_exp + leading_bit_index, so a_exp + b_exp must stay
            // below ~240 or the PE saturates to +/-inf (this format has no
            // subnormals and saturates rather than wrapping). [100,117] keeps the
            // sum <= 234, leaving headroom for the <=13-bit mantissa dot product.
            a_exp = mxint_exp_bits'(100 + ($urandom() % 18));   // [100,117]
            b_exp = mxint_exp_bits'(100 + ($urandom() % 18));

            dot = 0;
            for (int i = 0; i < group_size; i++) begin
                av = rand_mant();  bv = rand_mant();
                a_mant[i] = av;    b_mant[i] = bv;
                dot += longint'(av) * longint'(bv);
            end

            scale  = $pow(2.0, real'(a_exp) + real'(b_exp) - EXP_OFFSET);
            expval = real'(dot) * scale;

            // drive one block with a cleared accumulator
            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;

            if (acc_exp_o == 0 && acc_mant_o == 0)
                dutval = 0.0;
            else
                dutval = (acc_sign_o ? -1.0 : 1.0)
                       * (1.0 + real'(acc_mant_o) / $pow(2.0, real'(fp_mant_bits)))
                       * $pow(2.0, real'(acc_exp_o) - EXP_OFFSET);

            tol = 1e-9 * ((expval < 0.0) ? -expval : expval) + 1e-30;
            if (((dutval - expval) > tol) || ((expval - dutval) > tol)) begin
                fails++;
                if (fails <= 8)
                    $display("  FAIL[%0d] dot=%0d a_exp=%0d b_exp=%0d exp=%0.10g dut=%0.10g",
                             n, dot, a_exp, b_exp, expval, dutval);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
