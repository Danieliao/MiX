// Self-checking testbench for MiX_MX_Quantizer.
//
// The quantizer converts a 32-element FP32 block into either
//   * MiX  (sel_mix=1): shared_meta = {max_base_exp[4:0], shared_mant[2:0]},
//                       per element {sign, delta_e[2:0]}, and
//   * MX   (sel_mix=0): shared_meta = E8M0 exponent, per element MXINT4.
//
// Reference model below replays the RTL exactly (comparator tree -> rounder ->
// per-element refinement / asymmetric encode / MXINT shift), so the comparison
// is BIT-EXACT on both shared_meta and all 32 elements, for both formats.
module quantizer_tb;
    localparam int NUM_ELEMENTS = 32;
    localparam int FP32_BITS    = 32;
    localparam int MXINT_BITS   = 4;
    localparam int NVEC         = 1000;

    logic clk = 0, rst_n;
    logic sel_mix;
    logic [NUM_ELEMENTS-1:0][FP32_BITS-1:0] fp32_in;
    logic [7:0] shared_meta;
    logic [NUM_ELEMENTS-1:0][MXINT_BITS-1:0] quant_out;

    MiX_MX_Quantizer #(.NUM_ELEMENTS(NUM_ELEMENTS)) dut (
        .clk(clk), .rst_n(rst_n), .sel_mix(sel_mix),
        .fp32_in(fp32_in), .shared_meta(shared_meta), .quant_out(quant_out)
    );

    always #5 clk = ~clk;

    int seed = 32'hDEC0DE1;

    // reference outputs
    logic [7:0] ref_meta;
    logic [NUM_ELEMENTS-1:0][MXINT_BITS-1:0] ref_quant;

    task automatic reference(input logic sel);
        int sgn [NUM_ELEMENTS];
        int e   [NUM_ELEMENTS];
        int m4  [NUM_ELEMENTS];
        int m3  [NUM_ELEMENTS];
        int exp_max, mant4_max, key, best;
        int rounded, exp_r, mant_r;
        int man_diff, refinement, exp_diff_raw, delta_e;
        int mant_to_shift, shifted, mxv;
        begin
            // field extraction
            for (int i = 0; i < NUM_ELEMENTS; i++) begin
                sgn[i] = int'(fp32_in[i][31]);
                e[i]   = int'(fp32_in[i][30:23]);
                m4[i]  = int'(fp32_in[i][22:19]);
                m3[i]  = int'(fp32_in[i][22:20]);
            end

            // comparator tree: max magnitude by {exp, mant4}
            best = -1; exp_max = 0; mant4_max = 0;
            for (int i = 0; i < NUM_ELEMENTS; i++) begin
                key = (e[i] << 4) | m4[i];
                if (key > best) begin
                    best = key; exp_max = e[i]; mant4_max = m4[i];
                end
            end

            // rounder: {exp, mant4[3:1]} + mant4[0]
            rounded = ((exp_max << 3) | (m4_msb3(mant4_max))) + (mant4_max & 1);
            exp_r   = (rounded >> 3) & 8'hFF;
            mant_r  = rounded & 3'h7;

            ref_meta = sel ? 8'((exp_r & 5'h1F) << 3 | mant_r) : 8'(exp_r);

            // per-element
            for (int i = 0; i < NUM_ELEMENTS; i++) begin
                man_diff = mant_r - m3[i];
                if      (man_diff >=  4) refinement = -1;
                else if (man_diff <= -4) refinement =  1;
                else                     refinement =  0;

                exp_diff_raw = (exp_r - e[i]) & 8'hFF;
                delta_e      = exp_diff_raw + refinement;

                if (sel) begin
                    // MiX: asymmetric valid range (7 for positive, 6 for negative)
                    if (sgn[i] == 0)
                        ref_quant[i] = (delta_e > 7) ? 4'b1000 : {1'b0, 3'(delta_e & 7)};
                    else
                        ref_quant[i] = (delta_e > 6) ? 4'b1000 : {1'b1, 3'(delta_e & 7)};
                end else begin
                    // MX: shift {1, mant3[2:1]} right by the RAW exponent difference
                    mant_to_shift = 4 | ((m3[i] >> 1) & 3);
                    shifted = (exp_diff_raw >= 3) ? 0 : (mant_to_shift >> (exp_diff_raw & 3));
                    mxv = sgn[i] ? -shifted : shifted;
                    ref_quant[i] = 4'(mxv);
                end
            end
        end
    endtask

    function automatic int m4_msb3(input int m4);
        return (m4 >> 1) & 3'h7;
    endfunction

    initial begin
        int fails;
        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            sel_mix = n[0];   // alternate MiX / MX

            for (int i = 0; i < NUM_ELEMENTS; i++) begin
                // realistic FP32: exponent in a moderate band, random sign+mantissa
                fp32_in[i] = {logic'($urandom() % 2),
                              8'(100 + ($urandom() % 41)),
                              23'($urandom())};
            end

            reference(sel_mix);

            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); #1;   // single registered output stage

            if (shared_meta !== ref_meta || quant_out !== ref_quant) begin
                fails++;
                if (fails <= 5) begin
                    $display("  FAIL[%0d] sel_mix=%0b", n, sel_mix);
                    $display("     meta  ref=%h dut=%h", ref_meta, shared_meta);
                    $display("     quant ref=%h", ref_quant);
                    $display("           dut=%h", quant_out);
                end
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors BIT-EXACT (both MiX and MX)", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
