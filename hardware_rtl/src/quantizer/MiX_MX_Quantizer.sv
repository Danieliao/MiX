module MiX_MX_Quantizer #(
    parameter NUM_ELEMENTS = 32
    , parameter FP32_BITS = 32
    , parameter EXP_BITS = 8
    , parameter MAN_CMP_BITS = 4     // mantissa bits used for comparison
    , parameter MAN_BITS = 3         // mantissa bits after rounding
    , parameter MAX_BASE_EXP_BITS = 5
    , parameter INVMX_EXP_BITS = 3
    , parameter MXINT_BITS = 4
    , parameter LOG2_NUM = $clog2(NUM_ELEMENTS)
) (
    input logic clk
    , input logic rst_n

    // Format select: 1 = MiX output, 0 = MX (MXINT4) output
    , input logic sel_mix

    // Input: 32 FP32 values
    , input logic [NUM_ELEMENTS-1:0][FP32_BITS-1:0] fp32_in

    // Output: shared metadata (8 bits) + 32 × 4-bit quantized elements
    , output logic [7:0] shared_meta
    , output logic [NUM_ELEMENTS-1:0][MXINT_BITS-1:0] quant_out
);

    // ================================================================
    // Extract FP32 fields (reduced precision)
    // ================================================================
    logic [NUM_ELEMENTS-1:0] elem_sign;
    logic [NUM_ELEMENTS-1:0][EXP_BITS-1:0] elem_exp;
    logic [NUM_ELEMENTS-1:0][MAN_CMP_BITS-1:0] elem_mant4;  // MSB 4 bits
    logic [NUM_ELEMENTS-1:0][MAN_BITS-1:0] elem_mant3;       // MSB 3 bits

    always_comb begin
        for (int i = 0; i < NUM_ELEMENTS; i++) begin
            elem_sign[i]  = fp32_in[i][31];
            elem_exp[i]   = fp32_in[i][30:23];
            elem_mant4[i] = fp32_in[i][22:19];
            elem_mant3[i] = fp32_in[i][22:20];
        end
    end

    // ================================================================
    // Part 1: Comparator Tree — find max absolute value
    // ================================================================
    logic sign_max;
    logic [EXP_BITS-1:0] exp_max;
    logic [MAN_CMP_BITS-1:0] mant4_max;

    Quantizer_ComparatorTree #(
        .NUM_ELEMENTS(NUM_ELEMENTS)
        , .EXP_BITS(EXP_BITS)
        , .MAN_CMP_BITS(MAN_CMP_BITS)
    ) u_comp_tree (
        .sign_in(elem_sign)
        , .exp_in(elem_exp)
        , .mant4_in(elem_mant4)
        , .sign_max(sign_max)
        , .exp_max(exp_max)
        , .mant4_max(mant4_max)
    );

    // ================================================================
    // Part 2: Rounder — round 4b mantissa to 3b, possibly increment exp
    // ================================================================
    logic [EXP_BITS-1:0] exp_max_rounded;
    logic [MAN_BITS-1:0] mant3_max_rounded;

    Quantizer_Rounder #(
        .EXP_BITS(EXP_BITS)
        , .MAN_CMP_BITS(MAN_CMP_BITS)
        , .MAN_BITS(MAN_BITS)
    ) u_rounder (
        .exp_in(exp_max)
        , .mant4_in(mant4_max)
        , .exp_rounded(exp_max_rounded)
        , .mant_rounded(mant3_max_rounded)
    );

    // ================================================================
    // Parts 3-5: Per-element processing (32 instances)
    // ================================================================
    logic [NUM_ELEMENTS-1:0][MXINT_BITS-1:0] mix_out;
    logic signed [NUM_ELEMENTS-1:0][MXINT_BITS-1:0] mx_out;

    genvar i;
    generate
        for (i = 0; i < NUM_ELEMENTS; i++) begin : gen_per_elem
            Quantizer_PerElement #(
                .EXP_BITS(EXP_BITS)
                , .MAN_BITS(MAN_BITS)
                , .MAN_CMP_BITS(MAN_CMP_BITS)
                , .INVMX_EXP_BITS(INVMX_EXP_BITS)
                , .MXINT_BITS(MXINT_BITS)
            ) u_per_elem (
                .sign_i(elem_sign[i])
                , .exp_i(elem_exp[i])
                , .mant3_i(elem_mant3[i])
                , .exp_max_rounded(exp_max_rounded)
                , .mant3_max_rounded(mant3_max_rounded)
                , .mix_out(mix_out[i])
                , .mx_out(mx_out[i])
            );
        end
    endgenerate

    // ================================================================
    // Output MUX: select between MiX and MX formats
    // ================================================================
    // MiX shared metadata: {max_base_exp(5b), shared_mant(3b)}
    //   max_base_exp = EXP_max_rounded[4:0] (LSB 5 bits)
    //   shared_mant  = mant3_max_rounded[2:0]
    // MX shared metadata: {EXP_max_rounded(8b)}

    logic [7:0] mix_meta;
    logic [7:0] mx_meta;

    always_comb begin
        mix_meta = {exp_max_rounded[MAX_BASE_EXP_BITS-1:0], mant3_max_rounded};
        mx_meta  = exp_max_rounded;
    end

    // Registered output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shared_meta <= '0;
            quant_out   <= '0;
        end else begin
            shared_meta <= sel_mix ? mix_meta : mx_meta;
            for (int j = 0; j < NUM_ELEMENTS; j++) begin
                quant_out[j] <= sel_mix ? mix_out[j] : mx_out[j];
            end
        end
    end

endmodule
