// Self-checking testbench for MXFP4_g16_PE.
//
// Block-16 MXFP4 (E2M1 elements) packed into a 32-element PE: the E8M0 scale
// granularity stays 16 (still 4.5 b/element) and one PE consumes TWO 16-element
// sub-blocks, each with its own shared exponent.
//
//     value = SUM_g [ (SUM_{i in g} a_elem[i]*b_elem[i]) * 2^(a_exp[g]+b_exp[g]-254) ]
//
// E2M1 magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The PE normalizes each
// sub-block and folds both into the accumulator with one fused 3-input FP add;
// output decode:  value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 254).
module mxfp4_g16_tb;
    localparam int group_size      = 32;
    localparam int sub_group_size  = 16;
    localparam int NUM_SUB         = group_size / sub_group_size;   // 2
    localparam int shared_exp_bits = 8;
    localparam int elem_bits       = 4;
    localparam int fp_exp_bits     = 8;
    localparam int fp_mant_bits    = 23;
    localparam int NVEC            = 1000;
    localparam real EXP_OFFSET     = 254.0;   // 2 * E8M0 bias

    real LEVELS [0:7];

    logic clk = 0, rst_n, acc_shift = 0;

    logic [NUM_SUB-1:0][shared_exp_bits-1:0] a_exp, b_exp;
    logic [group_size-1:0][elem_bits-1:0] a_el, b_el;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [NUM_SUB-1:0][shared_exp_bits-1:0] a_exp_o, b_exp_o;
    logic [group_size-1:0][elem_bits-1:0] a_el_o, b_el_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MXFP4_g16_PE #(
        .group_size(group_size), .sub_group_size(sub_group_size),
        .shared_exp_bits(shared_exp_bits), .elem_bits(elem_bits),
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

    function automatic real e2m1_val(input logic [elem_bits-1:0] e);
        real m;
        m = LEVELS[e[2:0]];
        return e[3] ? -m : m;
    endfunction

    int seed = 32'h6BFA916;

    initial begin
        int fails;
        real expval, dutval, tol, absval, dot_g, sc;
        logic [elem_bits-1:0] ae, be;

        LEVELS[0]=0.0; LEVELS[1]=0.5; LEVELS[2]=1.0; LEVELS[3]=1.5;
        LEVELS[4]=2.0; LEVELS[5]=3.0; LEVELS[6]=4.0; LEVELS[7]=6.0;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            expval = 0.0;
            absval = 0.0;

            for (int g = 0; g < NUM_SUB; g++) begin
                // sum kept <= ~234 so the 8-bit accumulator exponent cannot saturate
                a_exp[g] = shared_exp_bits'(100 + ($urandom() % 18));
                b_exp[g] = shared_exp_bits'(100 + ($urandom() % 18));

                dot_g = 0.0;
                for (int i = 0; i < sub_group_size; i++) begin
                    ae = elem_bits'($urandom() % 16);
                    be = elem_bits'($urandom() % 16);
                    a_el[g*sub_group_size + i] = ae;
                    b_el[g*sub_group_size + i] = be;
                    dot_g += e2m1_val(ae) * e2m1_val(be);
                end

                sc = $pow(2.0, real'(a_exp[g]) + real'(b_exp[g]) - EXP_OFFSET);
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
