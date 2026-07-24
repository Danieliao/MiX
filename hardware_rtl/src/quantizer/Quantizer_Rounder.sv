module Quantizer_Rounder #(
    parameter EXP_BITS = 8
    , parameter MAN_CMP_BITS = 4    // 4-bit mantissa from comparator tree
    , parameter MAN_BITS = 3        // 3-bit mantissa after rounding
) (
    // Input: max element from comparator tree
    input logic [EXP_BITS-1:0] exp_in
    , input logic [MAN_CMP_BITS-1:0] mant4_in

    // Output: rounded {exp, mant_3b}
    , output logic [EXP_BITS-1:0] exp_rounded
    , output logic [MAN_BITS-1:0] mant_rounded
);

    // Round-to-nearest: add the 4th mantissa bit (rounding bit) to {exp, mant_msb3}
    // If mant_msb3 overflows (111 + 1 = 1000), the carry propagates into exponent
    logic [EXP_BITS+MAN_BITS-1:0] rounded;

    always_comb begin
        rounded = {exp_in, mant4_in[MAN_CMP_BITS-1:1]} + {{(EXP_BITS+MAN_BITS-1){1'b0}}, mant4_in[0]};
        exp_rounded  = rounded[EXP_BITS+MAN_BITS-1:MAN_BITS];
        mant_rounded = rounded[MAN_BITS-1:0];
    end

endmodule
