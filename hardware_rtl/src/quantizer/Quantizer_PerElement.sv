module Quantizer_PerElement #(
    parameter EXP_BITS = 8
    , parameter MAN_BITS = 3        // rounded mantissa bits
    , parameter MAN_CMP_BITS = 4    // mantissa bits for MX shift ({1, msb2})
    , parameter INVMX_EXP_BITS = 3  // MiX delta_e width
    , parameter MXINT_BITS = 4      // MXINT4 output width
) (
    // Per-element input
    input logic sign_i
    , input logic [EXP_BITS-1:0] exp_i
    , input logic [MAN_BITS-1:0] mant3_i       // MSB 3 bits of element's mantissa

    // Shared max (from rounder)
    , input logic [EXP_BITS-1:0] exp_max_rounded
    , input logic [MAN_BITS-1:0] mant3_max_rounded

    // MiX output
    , output logic [MXINT_BITS-1:0] mix_out     // {sign_i, delta_e[2:0]} or 4'b1000

    // MX (MXINT4) output
    , output logic signed [MXINT_BITS-1:0] mx_out  // 4-bit two's complement
);

    // ================================================================
    // Part 3: EXP Refinement
    // ================================================================

    // Mantissa difference: mant_max - mant_i (signed, range -7 to +7)
    logic signed [MAN_BITS:0] man_diff;

    // Refinement adjustment: -1, 0, or +1
    logic signed [1:0] refinement;

    // Raw exponent difference (before refinement)
    logic [EXP_BITS-1:0] exp_diff_raw;

    // Delta exponent (signed, after refinement)
    logic signed [EXP_BITS:0] delta_e;

    always_comb begin
        // Mantissa comparison: MAN_max - MAN_i
        man_diff = $signed({1'b0, mant3_max_rounded}) - $signed({1'b0, mant3_i});

        // Compare with ±0.5 (0.5 = 4 in 3-bit fractional representation)
        // > +0.5: max mantissa significantly larger → adjust -1
        // < -0.5: max mantissa significantly smaller → adjust +1
        if (man_diff >= 4'sd4)
            refinement = -2'sd1;
        else if (man_diff <= -4'sd4)
            refinement = 2'sd1;
        else
            refinement = 2'sd0;

        // Raw exponent difference
        exp_diff_raw = exp_max_rounded - exp_i;

        // Delta exponent with refinement
        delta_e = $signed({1'b0, exp_diff_raw}) + $signed({{(EXP_BITS-1){refinement[1]}}, refinement});
    end

    // ================================================================
    // Part 4: Asymmetric Quantization (MiX path)
    // ================================================================

    always_comb begin
        if (sign_i == 1'b0) begin
            // Positive: valid delta_e range [0, 7]
            if (delta_e > $signed(9'sd7))
                mix_out = 4'b1000;  // overflow → zero
            else
                mix_out = {1'b0, delta_e[INVMX_EXP_BITS-1:0]};
        end else begin
            // Negative: valid delta_e range [0, 6] (delta_e=0 → {1,000} = zero encoding)
            if (delta_e > $signed(9'sd6))
                mix_out = 4'b1000;  // overflow → zero
            else
                mix_out = {1'b1, delta_e[INVMX_EXP_BITS-1:0]};
        end
    end

    // ================================================================
    // Part 5: MX (MXINT4) Quantization Path
    // ================================================================

    // Use raw exponent difference (before refinement) to shift mantissa
    // Shift {1'b1, MAN_i[MSB 2 bits]} right by exp_diff_raw
    // Result is 3-bit unsigned, then convert to 4-bit signed with sign_i

    logic [2:0] mant_to_shift;   // {1, mant3_i[2:1]}
    logic [2:0] shifted_mant;
    logic signed [MXINT_BITS-1:0] mx_signed;

    always_comb begin
        // {implicit 1, top 2 mantissa bits} = 3-bit value
        mant_to_shift = {1'b1, mant3_i[MAN_BITS-1:MAN_BITS-2]};

        // Right-shift by exponent difference (large shift → 0)
        if (exp_diff_raw >= 8'd3)
            shifted_mant = 3'b0;
        else
            shifted_mant = mant_to_shift >> exp_diff_raw[1:0];

        // Apply sign: convert to 4-bit two's complement
        if (sign_i)
            mx_signed = -$signed({1'b0, shifted_mant});
        else
            mx_signed = $signed({1'b0, shifted_mant});

        mx_out = mx_signed;
    end

endmodule
