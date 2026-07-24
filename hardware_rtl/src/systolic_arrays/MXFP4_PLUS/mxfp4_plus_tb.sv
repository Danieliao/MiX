// Self-checking testbench for MXFP4_PLUS_PE.
//
// For each random block pair it (1) quantizes A and B per `_mxplus_q` — E2M1 for
// all elements, the largest-magnitude (BM) element as E0M3 (top-binade E2M3) with
// a 5-bit index — (2) accumulates the reference quantized dot product in `real`,
// (3) drives one block through the PE, and (4) decodes the FP32 output.
//
// Shared exponent uses a small bias of 15 (a_exp = se + 15) so the double-biased
// FP32 output exponent stays within 8 bits; decode is therefore 2^(exp - 30),
// same convention as amxfp4_tb. Output value (acc reset to 0):
//   value = (-1)^sign * (1 + mant/2^23) * 2^(exp - 30).
module mxfp4_plus_tb;
    localparam group_size  = 32;
    localparam shared_exp_bits = 8;
    localparam elem_bits   = 4;
    localparam bm_idx_bits = 5;
    localparam fp_exp_bits = 8;
    localparam fp_mant_bits = 23;
    localparam E2M1_MAX = 6.0;
    localparam EXP_BIAS = 15;        // small bias to avoid 8-bit FP32 exponent overflow
    localparam int NVEC = 600;

    real LEVELS [0:7];   // E2M1 positive levels
    real THRESH [0:6];

    logic clk = 0;
    logic rst_n;
    logic acc_shift;

    logic [shared_exp_bits-1:0] a_exp, b_exp;
    logic [group_size-1:0][elem_bits-1:0] a_el, b_el;
    logic [bm_idx_bits-1:0] a_bm, b_bm;

    logic acc_sign_in = 0;
    logic [fp_exp_bits-1:0] acc_exp_in = 0;
    logic [fp_mant_bits-1:0] acc_mant_in = 0;

    // Pass-through outputs (unused)
    logic [shared_exp_bits-1:0] a_exp_o, b_exp_o;
    logic [group_size-1:0][elem_bits-1:0] a_el_o, b_el_o;
    logic [bm_idx_bits-1:0] a_bm_o, b_bm_o;
    // Accumulator output
    logic acc_sign_o;
    logic [fp_exp_bits-1:0] acc_exp_o;
    logic [fp_mant_bits-1:0] acc_mant_o;

    MXFP4_PLUS_PE #(
        .group_size(group_size), .shared_exp_bits(shared_exp_bits), .elem_bits(elem_bits),
        .bm_idx_bits(bm_idx_bits), .fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .a_exp_in(a_exp), .a_elem_in(a_el), .a_bm_idx_in(a_bm),
        .b_exp_in(b_exp), .b_elem_in(b_el), .b_bm_idx_in(b_bm),
        .acc_sign_in(acc_sign_in), .acc_exp_in(acc_exp_in), .acc_mant_in(acc_mant_in),
        .a_exp_out(a_exp_o), .a_elem_out(a_el_o), .a_bm_idx_out(a_bm_o),
        .b_exp_out(b_exp_o), .b_elem_out(b_el_o), .b_bm_idx_out(b_bm_o),
        .acc_sign_out(acc_sign_o), .acc_exp_out(acc_exp_o), .acc_mant_out(acc_mant_o)
    );

    always #5 clk = ~clk;

    function automatic int e2m1_idx(input real a);
        real ac; int idx;
        ac = (a < 0.0) ? 0.0 : ((a > E2M1_MAX) ? E2M1_MAX : a);
        idx = 0;
        for (int k = 0; k < 7; k++) if (ac > THRESH[k]) idx = k + 1;
        return idx;
    endfunction

    // floor(log2(x)) for x > 0
    function automatic int flog2(input real x);
        return $floor($ln(x) / $ln(2.0) + 1e-9);
    endfunction

    // Quantize one block: fills elem bits, bm index, exponent field, and reconstructed reals.
    task automatic quant_block(input real v [group_size],
                               output logic [group_size-1:0][elem_bits-1:0] elem,
                               output logic [bm_idx_bits-1:0] bm_idx,
                               output logic [shared_exp_bits-1:0] exp_field,
                               output real recon [group_size]);
        real maxabs, mag, scaled, level, sc; int se, imax, idx, steps; logic sgn;
        maxabs = 1e-6; imax = 0;
        for (int i = 0; i < group_size; i++) begin
            mag = (v[i] < 0.0) ? -v[i] : v[i];
            if (mag > maxabs) begin maxabs = mag; imax = i; end
        end
        se = flog2(maxabs) - 2;          // floor shared exponent
        exp_field = se + EXP_BIAS;
        bm_idx = imax[bm_idx_bits-1:0];
        sc = $pow(2.0, real'(se));
        for (int i = 0; i < group_size; i++) begin
            sgn    = (v[i] < 0.0);
            scaled = ((sgn ? -v[i] : v[i])) / sc;     // |v|/2^se  (in [0, 8) ; ==[4,8) at BM)
            if (i == imax) begin
                // E0M3: top binade, value = 4*(1 + steps/8), steps 0..7
                steps = $rtoi((scaled / 4.0 - 1.0) * 8.0 + 0.5);
                if (steps < 0) steps = 0; if (steps > 7) steps = 7;
                elem[i]  = {sgn, steps[2:0]};
                level    = 4.0 * (1.0 + real'(steps) / 8.0);
            end else begin
                idx      = e2m1_idx(scaled);
                elem[i]  = {sgn, idx[2:0]};
                level    = LEVELS[idx];
            end
            recon[i] = (sgn ? -1.0 : 1.0) * level * sc;
        end
    endtask

    real va [group_size];
    real vb [group_size];
    real reca [group_size];
    real recb [group_size];
    int  seed = 32'h00ABCDEF;

    initial begin
        LEVELS[0]=0.0; LEVELS[1]=0.5; LEVELS[2]=1.0; LEVELS[3]=1.5;
        LEVELS[4]=2.0; LEVELS[5]=3.0; LEVELS[6]=4.0; LEVELS[7]=6.0;
        THRESH[0]=0.25; THRESH[1]=0.75; THRESH[2]=1.25; THRESH[3]=1.75;
        THRESH[4]=2.5;  THRESH[5]=3.5;  THRESH[6]=5.0;

        void'($urandom(seed));

        begin
            int fails, same_cnt; fails = 0; same_cnt = 0;
            for (int n = 0; n < NVEC; n++) begin
                automatic real expdot, absdot, dutval, tol;
                automatic int ia, ib, fa, fb;

                // random reals in ~[-8,8]; force a clear maximum so the BM is well-defined
                for (int i = 0; i < group_size; i++) begin
                    va[i] = real'((($urandom() % 16001)) - 8000) / 1000.0;
                    vb[i] = real'((($urandom() % 16001)) - 8000) / 1000.0;
                end
                fa = $urandom() % group_size;
                fb = $urandom() % group_size;
                va[fa] = ((($urandom() & 1) ? 1.0 : -1.0)) * (4.0 + real'($urandom() % 4000) / 1000.0);
                vb[fb] = ((($urandom() & 1) ? 1.0 : -1.0)) * (4.0 + real'($urandom() % 4000) / 1000.0);

                quant_block(va, a_el, a_bm, a_exp, reca);
                quant_block(vb, b_el, b_bm, b_exp, recb);
                ia = a_bm; ib = b_bm;
                if (ia == ib) same_cnt++;

                expdot = 0.0; absdot = 0.0;
                for (int i = 0; i < group_size; i++) begin
                    expdot += reca[i] * recb[i];
                    absdot += (reca[i] < 0.0 ? -reca[i] : reca[i]) * (recb[i] < 0.0 ? -recb[i] : recb[i]);
                end

                acc_shift = 0;
                rst_n = 0; @(posedge clk); #1;
                rst_n = 1;
                @(posedge clk);     // Stage 1: MUL
                @(posedge clk); #1; // Stage 2: normalize + accumulate (fused, MXFP4-style)

                if (acc_exp_o == 0 && acc_mant_o == 0)
                    dutval = 0.0;
                else
                    dutval = (acc_sign_o ? -1.0 : 1.0)
                           * (1.0 + real'(acc_mant_o) / $pow(2.0, 23.0))
                           * $pow(2.0, real'(acc_exp_o) - 2.0 * real'(EXP_BIAS));

                tol = 1e-3 * absdot + 1e-9;
                if (((dutval - expdot) > tol) || ((expdot - dutval) > tol)) begin
                    fails++;
                    if (fails <= 12)
                        $display("FAIL[%0d] ia=%0d ib=%0d: exp=%0.5f dut=%0.5f (absdot=%0.3f tol=%0.5f) exp_o=%0d mant_o=%h",
                                 n, ia, ib, expdot, dutval, absdot, tol, acc_exp_o, acc_mant_o);
                end
            end
            $display("(same-index BM cases: %0d / %0d)", same_cnt, NVEC);
            if (fails == 0) $display("PASS: all %0d vectors within tolerance", NVEC);
            else            $display("RESULT: %0d / %0d vectors FAILED", fails, NVEC);
        end
        $finish;
    end
endmodule
