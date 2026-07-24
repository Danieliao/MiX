// Self-checking testbench for MXFP4_PE (OCP MX FP4 / E2M1, block_size = 32).
//
// Format: one E8M0 shared exponent per 32-element block plus per-element E2M1
// values {sign, exp[1:0], mant} with exponent bias 1. The 8 representable
// magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
//
//     value = (SUM_i a_elem[i]*b_elem[i]) * 2^(a_exp-127) * 2^(b_exp-127)
//
// The MUL converts each E2M1 product to fixed point (2 fractional bits) and
// subtracts FIXED_SCALE = 2 from the combined exponent so the PE can reuse the
// MXINT4 normalizer. The accumulator exponent therefore carries the same double
// bias (2*127 = 254) and decodes as:
//     value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 254)
//
// All E2M1 products are exact multiples of 0.25 and the block sum fits well
// inside the 24-bit significand, so the comparison is exact up to an epsilon.
module mxfp4_tb;
    localparam int group_size      = 32;
    localparam int shared_exp_bits = 8;
    localparam int elem_bits       = 4;
    localparam int fp_exp_bits     = 8;
    localparam int fp_mant_bits    = 23;
    localparam int NVEC            = 1000;
    localparam real EXP_OFFSET     = 254.0;   // 2 * E8M0 bias

    // E2M1 magnitude for element magnitude field {exp[1:0], mant}
    real LEVELS [0:7];

    logic clk = 0, rst_n, acc_shift = 0;

    logic [shared_exp_bits-1:0] a_exp, b_exp;
    logic [group_size-1:0][elem_bits-1:0] a_el, b_el;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [shared_exp_bits-1:0] a_exp_o, b_exp_o;
    logic [group_size-1:0][elem_bits-1:0] a_el_o, b_el_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MXFP4_PE #(
        .group_size(group_size), .shared_exp_bits(shared_exp_bits),
        .elem_bits(elem_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .mxfp4_a_exp_in(a_exp), .mxfp4_a_elem_in(a_el),
        .mxfp4_b_exp_in(b_exp), .mxfp4_b_elem_in(b_el),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .mxfp4_a_exp_out(a_exp_o), .mxfp4_a_elem_out(a_el_o),
        .mxfp4_b_exp_out(b_exp_o), .mxfp4_b_elem_out(b_el_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    // decode one 4-bit E2M1 element to its real value
    function automatic real e2m1_val(input logic [elem_bits-1:0] e);
        real m;
        m = LEVELS[e[2:0]];
        return e[3] ? -m : m;
    endfunction

    int seed = 32'h4FA17C3;

    initial begin
        int fails;
        real expval, dutval, tol, scale, dot;
        logic [elem_bits-1:0] ae, be;

        LEVELS[0]=0.0; LEVELS[1]=0.5; LEVELS[2]=1.0; LEVELS[3]=1.5;
        LEVELS[4]=2.0; LEVELS[5]=3.0; LEVELS[6]=4.0; LEVELS[7]=6.0;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            // Keep a_exp + b_exp <= ~240: the 8-bit accumulator exponent field
            // holds a_exp + b_exp - 2 + leading_bit_index and saturates at 255
            // (no subnormals in this accumulator format).
            a_exp = shared_exp_bits'(100 + ($urandom() % 18));   // [100,117]
            b_exp = shared_exp_bits'(100 + ($urandom() % 18));

            dot = 0.0;
            for (int i = 0; i < group_size; i++) begin
                ae = elem_bits'($urandom() % 16);
                be = elem_bits'($urandom() % 16);
                a_el[i] = ae;  b_el[i] = be;
                dot += e2m1_val(ae) * e2m1_val(be);
            end

            scale  = $pow(2.0, real'(a_exp) + real'(b_exp) - EXP_OFFSET);
            expval = dot * scale;

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
                    $display("  FAIL[%0d] dot=%0.4f a_exp=%0d b_exp=%0d exp=%0.10g dut=%0.10g",
                             n, dot, a_exp, b_exp, expval, dutval);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
