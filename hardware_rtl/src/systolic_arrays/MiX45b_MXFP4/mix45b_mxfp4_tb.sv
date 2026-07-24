// Self-checking testbench for MiX45b_MXFP4_PE
// (MiX-4.5b INVMX activation x MXFP4 (E2M1) weight, block_size = 32).
//
// The 32-element block holds 2 weight groups of 16 (each with its own E8M0
// exponent) and 4 INVMX sub-groups of 8 (each with its own shared mantissa).
// The E2M1 weight exponent is absorbed into the per-element shift:
//
//   wgt_mant_full[i] = {exp_raw!=0, mant}                  (0..3)
//   wgt_exp[i]       = exp_raw==0 ? 0 : exp_raw - 1
//   signed_mant[i]   = invmx_exp[i]==7 ? 0 : +/- wgt_mant_full[i]
//                       (sign = invmx_sign ^ wgt_sign)
//   net_shift[i]     = MAX_MXFP4_EXP(2) - wgt_exp[i] + invmx_exp[i]
//   shifted[i]       = (signed_mant[i] << APPEND_ZEROS(7)) >>> net_shift[i]
//   sub_mult[g]      = (SUM of 8 shifted in sub-group g) * {1,mmm}[g]
//   wgt_sum[0]       = sub_mult[0] + sub_mult[1]   (elements  0-15)
//   wgt_sum[1]       = sub_mult[2] + sub_mult[3]   (elements 16-31)
//   out_exp[w]       = mxfp4_exp[w] + max_base_exp - TOTAL_FIXED_SCALE(9)
//
//   value = SUM_w wgt_sum[w] * 2^(out_exp[w] - 127)     (SINGLE E8M0 bias)
//
// The PE normalizes both weight-group results and folds them into the running
// accumulator with one fused 3-input FP add, hence the small tolerance.
module mix45b_mxfp4_tb;
    localparam int group_size        = 32;
    localparam int wgt_group_size    = 16;
    localparam int sub_group_size    = 8;
    localparam int NUM_WGT           = group_size / wgt_group_size;   // 2
    localparam int NUM_SUB           = group_size / sub_group_size;   // 4
    localparam int shared_exp_bits   = 8;
    localparam int elem_bits         = 4;
    localparam int max_base_exp_bits = 5;
    localparam int invmx_exp_bits    = 3;
    localparam int invmx_mant_bits   = 3;
    localparam int fp_exp_bits       = 8;
    localparam int fp_mant_bits      = 23;
    localparam int FRAC_BITS         = 5;    // hardcoded in the RTL
    localparam int MAX_MXFP4_EXP     = 2;
    localparam int APPEND_ZEROS      = FRAC_BITS + MAX_MXFP4_EXP;       // 7
    localparam int TOTAL_FIXED_SCALE = FRAC_BITS + 1 + invmx_mant_bits;  // 9
    localparam int NVEC              = 1000;
    localparam real EXP_OFFSET       = 127.0;

    logic clk = 0, rst_n, acc_shift = 0;

    logic [NUM_WGT-1:0][shared_exp_bits-1:0] mxfp4_exp;
    logic [group_size-1:0][elem_bits-1:0] mxfp4_elem;
    logic [max_base_exp_bits-1:0] max_base_exp;
    logic [group_size-1:0] invmx_sign;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp;
    logic [NUM_SUB-1:0][invmx_mant_bits-1:0] invmx_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [NUM_WGT-1:0][shared_exp_bits-1:0] mxfp4_exp_o;
    logic [group_size-1:0][elem_bits-1:0] mxfp4_elem_o;
    logic [max_base_exp_bits-1:0] max_base_exp_o;
    logic [group_size-1:0] invmx_sign_o;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_o;
    logic [NUM_SUB-1:0][invmx_mant_bits-1:0] invmx_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MiX45b_MXFP4_PE #(
        .group_size(group_size), .wgt_group_size(wgt_group_size),
        .sub_group_size(sub_group_size), .shared_exp_bits(shared_exp_bits),
        .elem_bits(elem_bits), .max_base_exp_bits(max_base_exp_bits),
        .invmx_exp_bits(invmx_exp_bits), .invmx_mant_bits(invmx_mant_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .mxfp4_exp_in(mxfp4_exp), .mxfp4_elem_in(mxfp4_elem),
        .invmx_max_base_exp_in(max_base_exp), .invmx_sign_in(invmx_sign),
        .invmx_exp_in(invmx_exp), .invmx_mant_in(invmx_mant),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .mxfp4_exp_out(mxfp4_exp_o), .mxfp4_elem_out(mxfp4_elem_o),
        .invmx_max_base_exp_out(max_base_exp_o), .invmx_sign_out(invmx_sign_o),
        .invmx_exp_out(invmx_exp_o), .invmx_mant_out(invmx_mant_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    int seed = 32'h45BFA40;

    initial begin
        int fails;
        int wsign, wexp_raw, wmant_raw, wexp, wfull, sm, nsh, sh;
        int sub_acc [NUM_SUB];
        longint sub_mult [NUM_SUB];
        longint wgt_sum;
        int cexp;
        real expval, dutval, absval, tol, sc;
        logic [elem_bits-1:0] we;
        logic [invmx_exp_bits-1:0] ie;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            max_base_exp = max_base_exp_bits'($urandom() % 32);
            for (int w = 0; w < NUM_WGT; w++)
                mxfp4_exp[w] = shared_exp_bits'(100 + ($urandom() % 41));
            for (int g = 0; g < NUM_SUB; g++)
                invmx_mant[g] = invmx_mant_bits'($urandom() % 8);

            for (int g = 0; g < NUM_SUB; g++) begin
                sub_acc[g] = 0;
                for (int j = 0; j < sub_group_size; j++) begin
                    automatic int i = g*sub_group_size + j;
                    we = elem_bits'($urandom() % 16);
                    ie = invmx_exp_bits'($urandom() % 8);
                    mxfp4_elem[i] = we;
                    invmx_exp[i]  = ie;
                    invmx_sign[i] = $urandom() % 2;

                    wsign     = int'(we[3]);
                    wexp_raw  = int'(we[2:1]);
                    wmant_raw = int'(we[0]);
                    wexp      = (wexp_raw == 0) ? 0 : (wexp_raw - 1);
                    wfull     = ((wexp_raw != 0) ? 2 : 0) + wmant_raw;

                    if (ie == 3'b111)                     sm = 0;
                    else if (int'(invmx_sign[i]) ^ wsign) sm = -wfull;
                    else                                  sm =  wfull;

                    nsh = MAX_MXFP4_EXP - wexp + int'(ie);
                    sh  = (sm <<< APPEND_ZEROS) >>> nsh;
                    sub_acc[g] += sh;
                end
                sub_mult[g] = longint'(sub_acc[g]) * longint'(8 + invmx_mant[g]);
            end

            expval = 0.0;
            absval = 0.0;
            for (int w = 0; w < NUM_WGT; w++) begin
                wgt_sum = sub_mult[2*w] + sub_mult[2*w + 1];
                cexp    = int'(mxfp4_exp[w]) + int'(max_base_exp) - TOTAL_FIXED_SCALE;
                sc      = $pow(2.0, real'(cexp) - EXP_OFFSET);
                expval += real'(wgt_sum) * sc;
                absval += ((wgt_sum < 0) ? real'(-wgt_sum) : real'(wgt_sum)) * sc;
            end

            // NOTE: this design keeps the normalize stage in its own pipeline
            // register (it is not the fused-backend variant), so the PE is
            // 3-stage: MUL -> normalize -> accumulate.
            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); @(posedge clk); #1;

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
