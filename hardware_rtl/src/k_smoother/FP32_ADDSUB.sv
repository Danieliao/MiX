// Combinational FP32 add/subtract on the {sign, 8b exp, 23b mant} format used by
// the systolic-array accumulators (normal bias, subnormals flushed, truncating
// round). Datapath reused from the PE FP32 accumulate (e.g. MXFP4_PE stage 2).
// y = a + b  (sub=0)   or   y = a - b  (sub=1)
module FP32_ADDSUB #(
    parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter W = 1 + fp_exp_bits + fp_mant_bits
) (
    input  logic [W-1:0] a
    , input  logic [W-1:0] b
    , input  logic sub
    , output logic [W-1:0] y
);

    logic a_sign, b_sign;
    logic [fp_exp_bits-1:0] a_exp, b_exp;
    logic [fp_mant_bits-1:0] a_mant, b_mant;

    logic [fp_exp_bits-1:0] exp_diff;
    logic a_exp_larger;
    logic [fp_exp_bits-1:0] max_exp;
    logic [fp_mant_bits:0] a_mant_ext, b_mant_ext;
    logic [fp_mant_bits:0] a_mant_aligned, b_mant_aligned;
    logic signed [fp_mant_bits+1:0] a_mant_signed, b_mant_signed;
    logic signed [fp_mant_bits+2:0] sum_mant;
    logic [fp_mant_bits+1:0] sum_mant_abs;
    logic y_sign;
    logic [fp_exp_bits-1:0] y_exp;
    logic [fp_mant_bits-1:0] y_mant;
    logic [fp_mant_bits:0] normalized_mant;
    logic [fp_exp_bits:0] normalized_exp;
    int add_lead_idx;

    always_comb begin
        a_sign = a[W-1];
        a_exp  = a[W-2 -: fp_exp_bits];
        a_mant = a[fp_mant_bits-1:0];
        b_sign = b[W-1] ^ sub;          // subtract = add with B's sign flipped
        b_exp  = b[W-2 -: fp_exp_bits];
        b_mant = b[fp_mant_bits-1:0];

        if (a_exp >= b_exp) begin
            exp_diff = a_exp - b_exp;
            a_exp_larger = 1'b1;
            max_exp = a_exp;
        end else begin
            exp_diff = b_exp - a_exp;
            a_exp_larger = 1'b0;
            max_exp = b_exp;
        end

        a_mant_ext = ((a_exp == '0) && (a_mant == '0)) ? '0 : {1'b1, a_mant};
        b_mant_ext = ((b_exp == '0) && (b_mant == '0)) ? '0 : {1'b1, b_mant};

        if (a_exp_larger) begin
            a_mant_aligned = a_mant_ext;
            b_mant_aligned = b_mant_ext >> exp_diff;
        end else begin
            a_mant_aligned = a_mant_ext >> exp_diff;
            b_mant_aligned = b_mant_ext;
        end

        a_mant_signed = a_sign ? -$signed({1'b0, a_mant_aligned}) : $signed({1'b0, a_mant_aligned});
        b_mant_signed = b_sign ? -$signed({1'b0, b_mant_aligned}) : $signed({1'b0, b_mant_aligned});
        sum_mant = a_mant_signed + b_mant_signed;

        if (sum_mant[fp_mant_bits+2]) begin
            y_sign = 1'b1;
            sum_mant_abs = (fp_mant_bits + 2)'(-sum_mant[fp_mant_bits+1:0]);
        end else begin
            y_sign = 1'b0;
            sum_mant_abs = sum_mant[fp_mant_bits+1:0];
        end

        add_lead_idx = -1;
        normalized_mant = '0;
        normalized_exp = '0;
        y_exp = '0;
        y_mant = '0;

        for (int i = fp_mant_bits + 1; i >= 0; i--) begin
            if ((add_lead_idx == -1) && sum_mant_abs[i]) begin
                add_lead_idx = i;
            end
        end

        if (add_lead_idx >= 0) begin
            if (add_lead_idx > fp_mant_bits) begin
                normalized_mant = {1'b0, sum_mant_abs} >> (add_lead_idx - fp_mant_bits);
            end else begin
                normalized_mant = (fp_mant_bits + 1)'(sum_mant_abs) << (fp_mant_bits - add_lead_idx);
            end

            if (add_lead_idx > fp_mant_bits) begin
                normalized_exp = (fp_exp_bits + 1)'(max_exp) + (add_lead_idx - fp_mant_bits);
            end else begin
                if (max_exp >= (fp_mant_bits - add_lead_idx)) begin
                    normalized_exp = (fp_exp_bits + 1)'(max_exp) - (fp_mant_bits - add_lead_idx);
                end else begin
                    normalized_exp = '0;
                end
            end

            if (normalized_exp[fp_exp_bits]) begin
                y_exp  = {fp_exp_bits{1'b1}};
                y_mant = {fp_mant_bits{1'b1}};
            end else begin
                y_exp  = normalized_exp[fp_exp_bits-1:0];
                y_mant = normalized_mant[fp_mant_bits-1:0];
            end
        end else begin
            y_sign = 1'b0;   // exact zero
        end

        y = {y_sign, y_exp, y_mant};
    end

endmodule
