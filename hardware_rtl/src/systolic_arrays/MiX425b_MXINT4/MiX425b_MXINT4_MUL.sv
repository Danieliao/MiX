module MiX425b_MXINT4_MUL #(
    parameter group_size = 32
    , parameter mxint_exp_bits = 8       // E8M0 shared exponent for MXINT4 weight
    , parameter mxint_mant_bits = 4      // MXINT4: sign + 3 mantissa (two's complement)
    , parameter max_base_exp_bits = 5    // INVMX E_max (5 bits)
    , parameter invmx_exp_bits = 3       // per-element exponent difference (0-6, 7=zero)
    , parameter invmx_mant_bits = 3      // shared mantissa (stored bits; full = {1, mmm})
    // Derived constants
    , parameter SIGNED_MANT_BITS = mxint_mant_bits  // 4 (input clamped to [-7,7], no MIN_INT negation overflow)
    , parameter SHIFTED_BITS = SIGNED_MANT_BITS + 3  // 7
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)     // 5
    , parameter ACC_BITS = SHIFTED_BITS + LOG2_GROUP_SIZE  // 12
    , parameter INVMX_FULL_MANT_BITS = invmx_mant_bits + 1  // 4 ({1, mmm})
    , parameter MULT_OUT_BITS = ACC_BITS + INVMX_FULL_MANT_BITS  // 16
    , parameter TOTAL_FIXED_SCALE = 3 + invmx_mant_bits  // 6
) (
    input logic clk
    , input logic rst_n

    // MXINT4 weight (shared exponent + per-element mantissas)
    , input logic [mxint_exp_bits-1:0] mxint_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant

    // INVMX activation (max_base_exp + per-element signs/exps + shared mantissa)
    , input logic [max_base_exp_bits-1:0] max_base_exp
    , input logic [group_size-1:0] invmx_sign
    , input logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp
    , input logic [invmx_mant_bits-1:0] invmx_mant

    // Output: combined exponent + signed multiply result
    , output logic [mxint_exp_bits:0] out_exp
    , output logic signed [MULT_OUT_BITS-1:0] out_sum
);

    // Step 1: Apply INVMX sign to MXINT mantissa
    // Widen to SIGNED_MANT_BITS to handle negation of MIN_INT (-8 → +8)
    logic signed [group_size-1:0][SIGNED_MANT_BITS-1:0] signed_mant;

    // Step 2: Append fractional zeros and right-shift by exponent difference
    logic signed [group_size-1:0][SHIFTED_BITS-1:0] shifted_mant;

    // Step 3: Adder tree result
    logic signed [ACC_BITS-1:0] acc_result;

    // Step 4: Multiply by shared mantissa {1, invmx_mant}
    logic signed [MULT_OUT_BITS-1:0] mult_result_pre;

    // Step 5: Combined exponent
    logic [mxint_exp_bits:0] combined_exp_pre;

    // ----------------------------------------------------------------
    // Step 1: Sign application (widen first to avoid MIN_INT overflow)
    // ----------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            if (invmx_exp[i] == 3'b111)  // zero value encoded by all-ones exponent difference
                signed_mant[i] = '0;
             else if (invmx_sign[i])
                signed_mant[i] = -SIGNED_MANT_BITS'(signed'(mxint_mant[i]));
            else
                signed_mant[i] =  SIGNED_MANT_BITS'(signed'(mxint_mant[i]));
        end
    end

    // ----------------------------------------------------------------
    // Step 2: Append 3 fractional zeros, then arithmetic right-shift.
    // Large invmx_exp naturally discards LSBs. All-ones invmx_exp encodes zero.
    // ----------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            shifted_mant[i] = $signed({signed_mant[i], {3{1'b0}}}) >>> invmx_exp[i];
        end
    end

    // ----------------------------------------------------------------
    // Step 3: Adder tree with per-stage 1-bit width growth
    // Stage s has width SHIFTED_BITS + s; final stage = ACC_BITS
    // ----------------------------------------------------------------
    genvar s, j;
    generate
        for (s = 0; s <= LOG2_GROUP_SIZE; s++) begin : tree
            localparam int W = SHIFTED_BITS + s;
            localparam int N = group_size >> s;
            logic signed [W-1:0] val [N-1:0];

            if (s == 0) begin : init
                for (j = 0; j < N; j++) begin : elem
                    assign val[j] = shifted_mant[j];
                end
            end else begin : reduce
                for (j = 0; j < N; j++) begin : elem
                    assign val[j] = W'(signed'(tree[s-1].val[2*j]))
                                  + W'(signed'(tree[s-1].val[2*j+1]));
                end
            end
        end
    endgenerate

    assign acc_result = tree[LOG2_GROUP_SIZE].val[0];

    // ----------------------------------------------------------------
    // Step 4: Single multiply by shared mantissa with implicit leading 1
    // {1, invmx_mant} = (1.mmm)_2, 4-bit unsigned (8-15)
    // Prepend 0 for signed multiply: {0, 1, mmm} = 5-bit signed positive
    // ----------------------------------------------------------------
    always_comb begin
        mult_result_pre = MULT_OUT_BITS'(signed'(acc_result))
                        * MULT_OUT_BITS'(signed'({1'b0, 1'b1, invmx_mant}));
    end

    // ----------------------------------------------------------------
    // Step 5: Combined exponent (subtract TOTAL_FIXED_SCALE so PE
    // reuses MXINT4-style normalization)
    // ----------------------------------------------------------------
    always_comb begin
        combined_exp_pre = (mxint_exp_bits + 1)'(mxint_exp)
                         + (mxint_exp_bits + 1)'(max_base_exp)
                         - (mxint_exp_bits + 1)'(TOTAL_FIXED_SCALE);
    end

    // ----------------------------------------------------------------
    // Register outputs
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_exp <= '0;
            out_sum <= '0;
        end else begin
            out_exp <= combined_exp_pre;
            out_sum <= mult_result_pre;
        end
    end

endmodule
