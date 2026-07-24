module MiX425b_MXFP4_MUL #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8    // E8M0 weight shared exponent
    , parameter elem_bits = 4          // E2M1 weight element
    , parameter max_base_exp_bits = 5  // INVMX E_max
    , parameter invmx_exp_bits = 3     // per-element exponent difference (0-6, 7=zero)
    , parameter invmx_mant_bits = 3    // shared mantissa (stored; full = {1, mmm})
    // E2M1 constants
    , parameter FP4_EXP_BITS = 2
    , parameter FP4_MANT_BITS = 1
    , parameter FP4_EXP_BIAS = 1
    , parameter FULL_MANT_BITS = FP4_MANT_BITS + 1     // 2 ({implicit, m})
    , parameter MAX_MXFP4_EXP = (1 << FP4_EXP_BITS) - 1 - FP4_EXP_BIAS  // 2
    // Derived constants
    , parameter SIGNED_MANT_BITS = FULL_MANT_BITS + 1   // 3 (sign + 2-bit mantissa, range ±3)
    , parameter APPEND_ZEROS = 5 + MAX_MXFP4_EXP  // 7
    , parameter SHIFTED_BITS = SIGNED_MANT_BITS + APPEND_ZEROS  // 10
    , parameter LOG2_GROUP_SIZE = $clog2(group_size)     // 5
    , parameter ACC_BITS = SHIFTED_BITS + LOG2_GROUP_SIZE  // 15
    , parameter INVMX_FULL_MANT_BITS = invmx_mant_bits + 1  // 4
    , parameter MULT_OUT_BITS = ACC_BITS + INVMX_FULL_MANT_BITS  // 19
    , parameter TOTAL_FIXED_SCALE = 5 + FP4_MANT_BITS + invmx_mant_bits  // 9
) (
    input logic clk
    , input logic rst_n

    // MXFP4 weight (E8M0 shared exponent + E2M1 per-element)
    , input logic [shared_exp_bits-1:0] mxfp4_exp
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_elem

    // INVMX activation (max_base_exp + per-element signs/exps + shared mantissa)
    , input logic [max_base_exp_bits-1:0] max_base_exp
    , input logic [group_size-1:0] invmx_sign
    , input logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp
    , input logic [invmx_mant_bits-1:0] invmx_mant

    // Output
    , output logic [shared_exp_bits:0] out_exp
    , output logic signed [MULT_OUT_BITS-1:0] out_sum
);

    // E2M1 decode signals
    logic [group_size-1:0] wgt_sign;
    logic [group_size-1:0][FP4_EXP_BITS-1:0] wgt_exp_raw;
    logic [group_size-1:0][FP4_MANT_BITS-1:0] wgt_mant_raw;
    logic [group_size-1:0][FP4_EXP_BITS-1:0] wgt_exp_decoded;
    logic [group_size-1:0][FULL_MANT_BITS-1:0] wgt_mant_full;

    // Combined sign and signed mantissa
    logic [group_size-1:0] combined_sign;
    logic signed [group_size-1:0][SIGNED_MANT_BITS-1:0] signed_mant;

    // Shifted values
    logic [group_size-1:0][3:0] net_right_shift;
    logic signed [group_size-1:0][SHIFTED_BITS-1:0] shifted_mant;

    // Adder tree result
    logic signed [ACC_BITS-1:0] acc_result;

    // Multiply result
    logic signed [MULT_OUT_BITS-1:0] mult_result_pre;

    // Combined exponent
    logic [shared_exp_bits:0] combined_exp_pre;

    // ----------------------------------------------------------------
    // Steps 1-2: Decode E2M1 weight, apply combined sign, compute shift
    // ----------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            // Decode E2M1: {sign, exp[1:0], mant[0]}
            wgt_sign[i]     = mxfp4_elem[i][elem_bits-1];
            wgt_exp_raw[i]  = mxfp4_elem[i][elem_bits-2:elem_bits-3];
            wgt_mant_raw[i] = mxfp4_elem[i][0];

            // Exponent: subtract bias; subnormal (exp_raw==0) stays 0
            wgt_exp_decoded[i] = (wgt_exp_raw[i] == '0) ? '0 : wgt_exp_raw[i] - FP4_EXP_BIAS[FP4_EXP_BITS-1:0];

            // Full mantissa with implicit bit (1 for normal, 0 for subnormal)
            wgt_mant_full[i] = {(wgt_exp_raw[i] != '0), wgt_mant_raw[i]};

            // Combined sign: INVMX activation sign XOR MXFP4 weight sign
            combined_sign[i] = invmx_sign[i] ^ wgt_sign[i];

            // Signed mantissa (3-bit signed, range ±3)
            
            if (invmx_exp[i] == 3'b111)  // zero value encoded by all-ones exponent difference
                signed_mant[i] = '0;
             else if (combined_sign[i])
                signed_mant[i] = -SIGNED_MANT_BITS'(wgt_mant_full[i]);
            else
                signed_mant[i] =  SIGNED_MANT_BITS'(wgt_mant_full[i]);

            // Append (5 + MAX_MXFP4_EXP) zeros, then right-shift by
            // net_right_shift = MAX_MXFP4_EXP - wgt_exp + invmx_exp (always >= 0).
            net_right_shift[i] = 4'(MAX_MXFP4_EXP) - 4'(wgt_exp_decoded[i]) + 4'(invmx_exp[i]);
            shifted_mant[i] = $signed({signed_mant[i], {APPEND_ZEROS{1'b0}}}) >>> net_right_shift[i];
        end
    end

    // ----------------------------------------------------------------
    // Step 3: Adder tree with per-stage 1-bit width growth
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
    // Step 4: Single multiply by shared INVMX mantissa with implicit leading 1
    // ----------------------------------------------------------------
    always_comb begin
        mult_result_pre = MULT_OUT_BITS'(signed'(acc_result))
                        * MULT_OUT_BITS'(signed'({1'b0, 1'b1, invmx_mant}));
    end

    // ----------------------------------------------------------------
    // Step 5: Combined exponent
    // TOTAL_FIXED_SCALE = 5 + FP4_MANT_BITS + invmx_mant_bits = 10
    // (extra FP4_MANT_BITS accounts for E2M1 mantissa fractional bit)
    // ----------------------------------------------------------------
    always_comb begin
        combined_exp_pre = (shared_exp_bits + 1)'(mxfp4_exp)
                         + (shared_exp_bits + 1)'(max_base_exp)
                         - (shared_exp_bits + 1)'(TOTAL_FIXED_SCALE);
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
