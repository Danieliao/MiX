// Combinational FP32 multiply on the {sign, 8b exp, 23b mant} format (normal bias,
// subnormals flushed, truncating round). Used to form mu = acc * (1/N).
//   y_exp = a_exp + b_exp + lead_idx - (2*fp_mant_bits + BIAS)
//   (derived: full = {1,mant} = 1.m * 2^fp_mant_bits, value = full * 2^(exp-BIAS-fp_mant_bits))
module FP32_MUL #(
    parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter W = 1 + fp_exp_bits + fp_mant_bits
) (
    input  logic [W-1:0] a
    , input  logic [W-1:0] b
    , output logic [W-1:0] y
);

    localparam int BIAS = (1 << (fp_exp_bits - 1)) - 1;          // 127
    localparam int FULL_MANT_W = fp_mant_bits + 1;               // 24
    localparam int PROD_W = 2 * FULL_MANT_W;                     // 48
    localparam int EXP_OFFSET = 2 * fp_mant_bits + BIAS;         // 173

    logic a_sign, b_sign;
    logic [fp_exp_bits-1:0] a_exp, b_exp;
    logic [fp_mant_bits-1:0] a_mant, b_mant;
    logic a_zero, b_zero;
    logic [FULL_MANT_W-1:0] a_full, b_full;
    logic [PROD_W-1:0] prod;
    int lead_idx;
    logic [fp_mant_bits:0] norm_mant;
    logic signed [fp_exp_bits+2:0] y_exp_signed;
    logic y_sign;
    logic [fp_exp_bits-1:0] y_exp;
    logic [fp_mant_bits-1:0] y_mant;

    always_comb begin
        a_sign = a[W-1];  a_exp = a[W-2 -: fp_exp_bits];  a_mant = a[fp_mant_bits-1:0];
        b_sign = b[W-1];  b_exp = b[W-2 -: fp_exp_bits];  b_mant = b[fp_mant_bits-1:0];

        a_zero = (a_exp == '0) && (a_mant == '0);
        b_zero = (b_exp == '0) && (b_mant == '0);

        a_full = a_zero ? '0 : {1'b1, a_mant};
        b_full = b_zero ? '0 : {1'b1, b_mant};

        prod   = a_full * b_full;
        y_sign = a_sign ^ b_sign;

        lead_idx = -1;
        norm_mant = '0;
        y_exp_signed = '0;
        y_exp = '0;
        y_mant = '0;

        for (int i = PROD_W - 1; i >= 0; i--) begin
            if ((lead_idx == -1) && prod[i]) begin
                lead_idx = i;
            end
        end

        if (lead_idx >= 0) begin
            // lead_idx >= 2*fp_mant_bits for any nonzero product → always shift right
            if (lead_idx >= fp_mant_bits) begin
                norm_mant = prod >> (lead_idx - fp_mant_bits);
            end else begin
                norm_mant = (fp_mant_bits + 1)'(prod) << (fp_mant_bits - lead_idx);
            end

            y_exp_signed = (fp_exp_bits+3)'(a_exp) + (fp_exp_bits+3)'(b_exp)
                         + (fp_exp_bits+3)'(lead_idx) - (fp_exp_bits+3)'(EXP_OFFSET);

            if (y_exp_signed <= 0) begin
                y_exp  = '0;          // underflow → flush to zero
                y_mant = '0;
                y_sign = 1'b0;
            end else if (y_exp_signed >= (1 << fp_exp_bits) - 1) begin
                y_exp  = {fp_exp_bits{1'b1}};    // overflow → saturate
                y_mant = {fp_mant_bits{1'b1}};
            end else begin
                y_exp  = y_exp_signed[fp_exp_bits-1:0];
                y_mant = norm_mant[fp_mant_bits-1:0];
            end
        end

        y = {y_sign, y_exp, y_mant};
    end

endmodule
