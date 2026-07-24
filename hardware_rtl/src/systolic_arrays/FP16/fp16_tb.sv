// Self-checking testbench for FP16_PE (IEEE half-precision baseline).
//
// This is the dense FP16 reference array: each PE holds ONE multiplier
// (group_size = 1) and accumulates into the custom FP32 accumulator, so the
// 512-MAC/cycle configuration is a 16x32 (or 32x32 halved) mesh rather than the
// 4x4 mesh used by the block-quantized formats.
//
// FP16 = {sign(1), exp(5), mant(10)}, bias 15, value = (-1)^s * (1.m) * 2^(exp-15).
// The MUL emits the raw exponent SUM and the 11x11 -> 22-bit mantissa product
// without removing any bias, so the accumulator exponent carries an implicit
// bias of 2*15 + 20 = 50:
//   value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 50)
//
// Stimulus uses NORMAL FP16 only (exp in [1,30]); subnormals, Inf and NaN are
// out of scope for this datapath (the accumulator has no subnormal support).
module fp16_tb;
    localparam int fp16_bits      = 16;
    localparam int fp16_exp_bits  = 5;
    localparam int fp16_mant_bits = 10;
    localparam int fp_exp_bits    = 8;
    localparam int fp_mant_bits   = 23;
    localparam int NVEC           = 1000;
    localparam real EXP_OFFSET    = 50.0;   // 2*15 (FP16 bias) + 20 (mantissa bits)

    logic clk = 0, rst_n, acc_shift = 0;

    logic [fp16_bits-1:0] fp16_a, fp16_b;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [fp16_bits-1:0] fp16_a_o, fp16_b_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    FP16_PE #(
        .fp16_bits(fp16_bits), .fp16_exp_bits(fp16_exp_bits),
        .fp16_mant_bits(fp16_mant_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .fp16_a_in(fp16_a), .fp16_b_in(fp16_b),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .fp16_a_out(fp16_a_o), .fp16_b_out(fp16_b_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    // decode a NORMAL FP16 bit pattern to real
    function automatic real fp16_val(input logic [fp16_bits-1:0] h);
        int e, m;
        real v;
        e = int'(h[14:10]);
        m = int'(h[9:0]);
        v = (1.0 + real'(m) / 1024.0) * $pow(2.0, real'(e) - 15.0);
        return h[15] ? -v : v;
    endfunction

    int seed = 32'h0FF16AB;

    initial begin
        int fails;
        real expval, dutval, tol;
        logic [4:0] ea, eb;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            ea = 5'(1 + ($urandom() % 30));   // [1,30] normal
            eb = 5'(1 + ($urandom() % 30));
            fp16_a = {logic'($urandom() % 2), ea, 10'($urandom() % 1024)};
            fp16_b = {logic'($urandom() % 2), eb, 10'($urandom() % 1024)};

            expval = fp16_val(fp16_a) * fp16_val(fp16_b);

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
                    $display("  FAIL[%0d] a=%h b=%h exp=%0.10g dut=%0.10g",
                             n, fp16_a, fp16_b, expval, dutval);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
