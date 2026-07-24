// Self-checking testbench for MiX425b_MXFP4_PE
// (MiX-4.25b INVMX activation x MXFP4 (E2M1) weight, block_size = 32).
//
// The INVMX shared mantissa factors out of the dot product, so each element only
// needs a shift-and-add. Because the weight is E2M1 (not an integer mantissa),
// the per-element shift also absorbs the weight's own exponent:
//
//   wgt_mant_full[i] = {exp_raw!=0, mant}                 (0..3)
//   wgt_exp[i]       = exp_raw==0 ? 0 : exp_raw - 1       (E2M1 bias 1)
//   signed_mant[i]   = invmx_exp[i]==7 ? 0
//                                      : +/- wgt_mant_full[i]   (sign = invmx_sign ^ wgt_sign)
//   net_shift[i]     = MAX_MXFP4_EXP(2) - wgt_exp[i] + invmx_exp[i]     (always >= 0)
//   shifted[i]       = (signed_mant[i] << APPEND_ZEROS(7)) >>> net_shift[i]
//   out_sum          = (SUM_i shifted[i]) * {1,mmm}       (8..15)
//   out_exp          = mxfp4_exp + max_base_exp - TOTAL_FIXED_SCALE(9)
//
//   value = out_sum * 2^(out_exp - 127)
//
// Only the weight side carries an E8M0 bias, so the accumulator decodes with a
// SINGLE bias: value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 127).
// The reference below replays the same integer ops, so the match is EXACT.
module mix425b_mxfp4_tb;
    localparam int group_size        = 32;
    localparam int shared_exp_bits   = 8;
    localparam int elem_bits         = 4;
    localparam int max_base_exp_bits = 5;
    localparam int invmx_exp_bits    = 3;
    localparam int invmx_mant_bits   = 3;
    localparam int fp_exp_bits       = 8;
    localparam int fp_mant_bits      = 23;
    localparam int FRAC_BITS         = 5;    // hardcoded in the RTL
    localparam int MAX_MXFP4_EXP     = 2;
    localparam int APPEND_ZEROS      = FRAC_BITS + MAX_MXFP4_EXP;      // 7
    localparam int TOTAL_FIXED_SCALE = FRAC_BITS + 1 + invmx_mant_bits; // 9
    localparam int NVEC              = 1000;
    localparam real EXP_OFFSET       = 127.0;

    logic clk = 0, rst_n, acc_shift = 0;

    logic [shared_exp_bits-1:0] mxfp4_exp;
    logic [group_size-1:0][elem_bits-1:0] mxfp4_elem;
    logic [max_base_exp_bits-1:0] max_base_exp;
    logic [group_size-1:0] invmx_sign;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp;
    logic [invmx_mant_bits-1:0] invmx_mant;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0]  acc_exp_in  = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    logic [shared_exp_bits-1:0] mxfp4_exp_o;
    logic [group_size-1:0][elem_bits-1:0] mxfp4_elem_o;
    logic [max_base_exp_bits-1:0] max_base_exp_o;
    logic [group_size-1:0] invmx_sign_o;
    logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_o;
    logic [invmx_mant_bits-1:0] invmx_mant_o;
    logic acc_sign_o;
    logic [fp_exp_bits-1:0]  acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MiX425b_MXFP4_PE #(
        .group_size(group_size), .shared_exp_bits(shared_exp_bits),
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

    int seed = 32'h425BFA4;

    initial begin
        int fails;
        int wsign, wexp_raw, wmant_raw, wexp, wfull, sm, nsh, sh, acc_int;
        longint out_sum_ref;
        int cexp;
        real expval, dutval, tol;
        logic [elem_bits-1:0] we;
        logic [invmx_exp_bits-1:0] ie;

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            mxfp4_exp    = shared_exp_bits'(100 + ($urandom() % 41));   // [100,140]
            max_base_exp = max_base_exp_bits'($urandom() % 32);
            invmx_mant   = invmx_mant_bits'($urandom() % 8);

            acc_int = 0;
            for (int i = 0; i < group_size; i++) begin
                we = elem_bits'($urandom() % 16);
                ie = invmx_exp_bits'($urandom() % 8);   // 7 = zero element
                mxfp4_elem[i] = we;
                invmx_exp[i]  = ie;
                invmx_sign[i] = $urandom() % 2;

                // --- reference: identical integer ops to the RTL ---
                wsign     = int'(we[3]);
                wexp_raw  = int'(we[2:1]);
                wmant_raw = int'(we[0]);
                wexp      = (wexp_raw == 0) ? 0 : (wexp_raw - 1);
                wfull     = ((wexp_raw != 0) ? 2 : 0) + wmant_raw;

                if (ie == 3'b111)                      sm = 0;
                else if (int'(invmx_sign[i]) ^ wsign)  sm = -wfull;
                else                                   sm =  wfull;

                nsh = MAX_MXFP4_EXP - wexp + int'(ie);   // always >= 0
                sh  = (sm <<< APPEND_ZEROS) >>> nsh;     // arithmetic (truncating)
                acc_int += sh;
            end

            out_sum_ref = longint'(acc_int) * longint'(8 + invmx_mant);
            cexp        = int'(mxfp4_exp) + int'(max_base_exp) - TOTAL_FIXED_SCALE;
            expval      = real'(out_sum_ref) * $pow(2.0, real'(cexp) - EXP_OFFSET);

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
                             n, out_sum_ref, cexp, expval, dutval);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors EXACT match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
