// Self-checking testbench for NVFP4_PE
// (NVIDIA Blackwell FP4: E2M1 elements + E4M3 block scale, block_size = 16).
//
// NVFP4's native scale granularity is 16, so a 32-wide PE carries TWO scales per
// operand; each 16-element sub-block has its own E4M3 scale (still 4.5 b/element).
//
//   value = SUM_g [ (SUM_{i in g} a_elem[i]*b_elem[i]) * scale_a[g] * scale_b[g] ]
//
// E2M1 magnitudes are {0,0.5,1,1.5,2,3,4,6}. An E4M3 scale field is
// {exp[6:3], mant[2:0]} (no sign) with exponent bias 7, so a normal scale is
//   scale = ({1,mant}/8) * 2^(exp-7).
//
// The MUL emits, per sub-block, the fixed-point element dot product, the scale
// exponent sum, and the 8-bit scale-mantissa product. The PE multiplies them,
// normalizes, and subtracts TOTAL_FIXED_SCALE = 2 + 2*3 = 8, leaving the
// accumulator exponent double-biased by 2*7 = 14:
//   value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 14)
//
// Scale exponents are drawn from the NORMAL range [5,14]: exp == 0 would be an
// E4M3 subnormal, which this datapath does not special-case (block scales are
// never subnormal in practice).
module nvfp4_tb;
    localparam int group_size     = 32;
    localparam int sub_group_size = 16;
    localparam int NUM_SUB        = group_size / sub_group_size;   // 2
    localparam int scale_bits     = 8;
    localparam int elem_bits      = 4;
    localparam int fp_exp_bits    = 8;
    localparam int fp_mant_bits   = 23;
    localparam int NVEC           = 1000;
    localparam real EXP_OFFSET    = 14.0;   // 2 * E4M3 bias

    real LEVELS [0:7];

    logic clk = 0, rst_n, acc_shift = 0;

    logic [NUM_SUB-1:0][scale_bits-1:0] a_scale, b_scale;
    logic [group_size-1:0][elem_bits-1:0] a_el, b_el;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [NUM_SUB-1:0][scale_bits-1:0] a_scale_o, b_scale_o;
    logic [group_size-1:0][elem_bits-1:0] a_el_o, b_el_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    NVFP4_PE #(
        .group_size(group_size), .sub_group_size(sub_group_size),
        .scale_bits(scale_bits), .elem_bits(elem_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .nvfp4_a_scale_in(a_scale), .nvfp4_a_elem_in(a_el),
        .nvfp4_b_scale_in(b_scale), .nvfp4_b_elem_in(b_el),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .nvfp4_a_scale_out(a_scale_o), .nvfp4_a_elem_out(a_el_o),
        .nvfp4_b_scale_out(b_scale_o), .nvfp4_b_elem_out(b_el_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    function automatic real e2m1_val(input logic [elem_bits-1:0] e);
        real m;
        m = LEVELS[e[2:0]];
        return e[3] ? -m : m;
    endfunction

    // E4M3 scale (normal range only): ({1,mant}/8) * 2^(exp-7)
    function automatic real e4m3_val(input logic [scale_bits-1:0] s);
        int e, m;
        e = int'(s[6:3]);
        m = int'(s[2:0]);
        return (real'(8 + m) / 8.0) * $pow(2.0, real'(e) - 7.0);
    endfunction

    int seed = 32'h07FA401;

    initial begin
        int fails;
        real expval, dutval, absval, tol, dot_g, sc;
        logic [elem_bits-1:0] ae, be;
        logic [3:0] ea, eb;
        logic [2:0] ma, mb;

        LEVELS[0]=0.0; LEVELS[1]=0.5; LEVELS[2]=1.0; LEVELS[3]=1.5;
        LEVELS[4]=2.0; LEVELS[5]=3.0; LEVELS[6]=4.0; LEVELS[7]=6.0;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            expval = 0.0;
            absval = 0.0;

            for (int g = 0; g < NUM_SUB; g++) begin
                ea = 4'(5 + ($urandom() % 10));   // [5,14] normal E4M3 exponent
                eb = 4'(5 + ($urandom() % 10));
                ma = 3'($urandom() % 8);
                mb = 3'($urandom() % 8);
                a_scale[g] = {1'b0, ea, ma};
                b_scale[g] = {1'b0, eb, mb};

                dot_g = 0.0;
                for (int i = 0; i < sub_group_size; i++) begin
                    ae = elem_bits'($urandom() % 16);
                    be = elem_bits'($urandom() % 16);
                    a_el[g*sub_group_size + i] = ae;
                    b_el[g*sub_group_size + i] = be;
                    dot_g += e2m1_val(ae) * e2m1_val(be);
                end

                sc = e4m3_val(a_scale[g]) * e4m3_val(b_scale[g]);
                expval += dot_g * sc;
                absval += ((dot_g < 0.0) ? -dot_g : dot_g) * sc;
            end

            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;

            if (acc_exp_o == 0 && acc_mant_o == 0)
                dutval = 0.0;
            else
                dutval = (acc_sign_o ? -1.0 : 1.0)
                       * (1.0 + real'(acc_mant_o) / $pow(2.0, real'(fp_mant_bits)))
                       * $pow(2.0, real'(acc_exp_o) - EXP_OFFSET);

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
