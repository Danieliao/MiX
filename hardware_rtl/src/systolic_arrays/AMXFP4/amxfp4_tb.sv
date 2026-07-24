// Self-checking testbench for AMXFP4_PE.
//
// For each random block pair it (1) quantizes A and B exactly as `_amxfp4_q`
// (two E5M2 scales per block, one per sign), (2) accumulates the reference
// quantized dot product in `real`, (3) drives one block through the PE, and
// (4) decodes the PE's custom double-bias FP32 output and compares.
//
// Output decode (single block, accumulator reset to 0):
//   value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 30),   30 = 2*E5M2_bias.
module amxfp4_tb;
    localparam group_size  = 32;
    localparam scale_bits  = 7;
    localparam elem_bits   = 4;
    localparam fp_exp_bits = 8;
    localparam fp_mant_bits = 23;
    localparam E2M1_MAX = 6.0;
    localparam int NVEC = 500;

    // E2M1 representable levels (positive)
    real LEVELS [0:7];
    real THRESH [0:6];

    logic clk = 0;
    logic rst_n;
    logic acc_shift;

    logic [scale_bits-1:0] a_sp, a_sn, b_sp, b_sn;
    logic [group_size-1:0][elem_bits-1:0] a_el, b_el;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0] acc_exp_in = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    // Pass-through outputs (unused)
    logic [scale_bits-1:0] a_sp_o, a_sn_o, b_sp_o, b_sn_o;
    logic [group_size-1:0][elem_bits-1:0] a_el_o, b_el_o;
    // Accumulator output
    logic acc_sign_o;
    logic [fp_exp_bits-1:0] acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    AMXFP4_PE #(
        .group_size(group_size), .scale_bits(scale_bits), .elem_bits(elem_bits),
        .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .amxfp4_a_scale_pos_in(a_sp), .amxfp4_a_scale_neg_in(a_sn), .amxfp4_a_elem_in(a_el),
        .amxfp4_b_scale_pos_in(b_sp), .amxfp4_b_scale_neg_in(b_sn), .amxfp4_b_elem_in(b_el),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .amxfp4_a_scale_pos_out(a_sp_o), .amxfp4_a_scale_neg_out(a_sn_o), .amxfp4_a_elem_out(a_el_o),
        .amxfp4_b_scale_pos_out(b_sp_o), .amxfp4_b_scale_neg_out(b_sn_o), .amxfp4_b_elem_out(b_el_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    // --- E2M1 quantization (returns index 0..7) ---
    function automatic int e2m1_idx(input real a);
        real ac; int idx;
        ac = (a < 0.0) ? 0.0 : ((a > E2M1_MAX) ? E2M1_MAX : a);
        idx = 0;
        for (int k = 0; k < 7; k++) if (ac > THRESH[k]) idx = k + 1;
        return idx;
    endfunction

    // --- E5M2 encode of a positive scale: outputs bit-field + reconstructed real ---
    task automatic e5m2(input real s, output int exp_field, output int mant_field, output real recon);
        real l2, m; int e, steps;
        if (s < 1e-6) s = 1e-6;
        l2 = $ln(s) / $ln(2.0);
        e  = $floor(l2 + 1e-9);
        m  = s / $pow(2.0, real'(e));        // [1,2)
        steps = $rtoi((m - 1.0) * 4.0 + 0.5); // round to 0..4
        if (steps >= 4) begin e = e + 1; steps = 0; end
        exp_field  = e + 15;                  // bias 15
        mant_field = steps;
        recon = (1.0 + real'(steps) / 4.0) * $pow(2.0, real'(e));
    endtask

    // Reference (quantized) reconstructed value of one element under a sign-matched scale
    task automatic quant_elem(input real v, input real sp, input real sn,
                              output logic [3:0] bits, output real recon);
        logic sign; real mag, sc; int idx;
        sign = (v < 0.0);
        mag  = sign ? -v : v;
        sc   = sign ? sn : sp;
        idx  = e2m1_idx(mag / sc);
        bits = {sign, idx[2:0]};
        recon = (sign ? -1.0 : 1.0) * LEVELS[idx] * sc;
    endtask

    real va [0:group_size-1];
    real vb [0:group_size-1];
    int  seed = 32'hC0FFEE;

    initial begin
        LEVELS[0]=0.0; LEVELS[1]=0.5; LEVELS[2]=1.0; LEVELS[3]=1.5;
        LEVELS[4]=2.0; LEVELS[5]=3.0; LEVELS[6]=4.0; LEVELS[7]=6.0;
        THRESH[0]=0.25; THRESH[1]=0.75; THRESH[2]=1.25; THRESH[3]=1.75;
        THRESH[4]=2.5;  THRESH[5]=3.5;  THRESH[6]=5.0;

        void'($urandom(seed));  // seed the global RNG once

        begin
            int fails; fails = 0;
            for (int n = 0; n < NVEC; n++) begin
                automatic real pmax_a, nmax_a, pmax_b, nmax_b;
                automatic int  ef; automatic int mf;
                automatic real sap, san, sbp, sbn;
                automatic real ra, rb, expdot, absdot, dutval, tol;
                automatic logic [3:0] bits;
                automatic int ia, ib;

                // random reals in ~[-8,8]
                pmax_a = 1e-6; nmax_a = 1e-6; pmax_b = 1e-6; nmax_b = 1e-6;
                for (int i = 0; i < group_size; i++) begin
                    ia = $urandom() % 16001;  // signed int: 0..16000
                    ib = $urandom() % 16001;
                    va[i] = real'(ia - 8000) / 1000.0;
                    vb[i] = real'(ib - 8000) / 1000.0;
                    if (va[i] > 0.0 && va[i] > pmax_a) pmax_a = va[i];
                    if (va[i] < 0.0 && -va[i] > nmax_a) nmax_a = -va[i];
                    if (vb[i] > 0.0 && vb[i] > pmax_b) pmax_b = vb[i];
                    if (vb[i] < 0.0 && -vb[i] > nmax_b) nmax_b = -vb[i];
                end

                // two E5M2 scales per block (scale = max/E2M1_MAX)
                e5m2(pmax_a / E2M1_MAX, ef, mf, sap); a_sp = {ef[4:0], mf[1:0]};
                e5m2(nmax_a / E2M1_MAX, ef, mf, san); a_sn = {ef[4:0], mf[1:0]};
                e5m2(pmax_b / E2M1_MAX, ef, mf, sbp); b_sp = {ef[4:0], mf[1:0]};
                e5m2(nmax_b / E2M1_MAX, ef, mf, sbn); b_sn = {ef[4:0], mf[1:0]};

                // quantize elements, pack bits, accumulate reference dot product
                expdot = 0.0; absdot = 0.0;
                for (int i = 0; i < group_size; i++) begin
                    quant_elem(va[i], sap, san, bits, ra); a_el[i] = bits;
                    quant_elem(vb[i], sbp, sbn, bits, rb); b_el[i] = bits;
                    expdot += ra * rb;
                    absdot += (ra < 0.0 ? -ra : ra) * (rb < 0.0 ? -rb : rb);
                end

                // drive one block with a clean (reset) accumulator
                acc_shift = 0;
                rst_n = 0; @(posedge clk); #1;
                rst_n = 1;
                @(posedge clk);   // Stage 1: MUL registers
                @(posedge clk);   // Stage 2: block-result registers
                @(posedge clk); #1; // Stage 3: accumulator registers the block product

                if (acc_exp_o == 0 && acc_mant_o == 0)
                    dutval = 0.0;
                else
                    dutval = (acc_sign_o ? -1.0 : 1.0)
                           * (1.0 + real'(acc_mant_o) / $pow(2.0, 23.0))
                           * $pow(2.0, real'(acc_exp_o) - 30.0);

                tol = 1e-3 * absdot + 1e-9;
                if (((dutval - expdot) > tol) || ((expdot - dutval) > tol)) begin
                    fails++;
                    if (fails <= 10)
                        $display("FAIL[%0d]: exp=%0.6f dut=%0.6f (absdot=%0.4f tol=%0.6f) | exp_o=%0d mant_o=%h sign=%b",
                                 n, expdot, dutval, absdot, tol, acc_exp_o, acc_mant_o, acc_sign_o);
                end
            end
            if (fails == 0)
                $display("PASS: all %0d vectors within tolerance", NVEC);
            else
                $display("RESULT: %0d / %0d vectors FAILED", fails, NVEC);
        end
        $finish;
    end
endmodule
