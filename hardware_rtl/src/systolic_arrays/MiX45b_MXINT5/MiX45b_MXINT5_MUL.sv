module MiX45b_MXINT5_MUL #(
    parameter group_size = 32           // MiX activation group size (elements per PE)
    , parameter wgt_group_size = 16     // MXINT4 weight group size
    , parameter sub_group_size = 8      // MiX mantissa sub-group size
    , parameter mxint_exp_bits = 8      // E8M0 shared exponent
    , parameter mxint_mant_bits = 5     // MXINT5: sign + 4 mantissa
    , parameter max_base_exp_bits = 5   // INVMX E_max
    , parameter invmx_exp_bits = 3      // per-element exponent difference (0-6, 7=zero)
    , parameter invmx_mant_bits = 3     // sub-group mantissa (stored; full = {1, mmm})
    // Derived constants
    , parameter NUM_WGT_GROUPS = group_size / wgt_group_size   // 2
    , parameter NUM_SUB_GROUPS = group_size / sub_group_size   // 4
    , parameter SUBS_PER_WGT = wgt_group_size / sub_group_size // 2
    , parameter SIGNED_MANT_BITS = mxint_mant_bits  // 5 (input clamped to [-15,15])
    , parameter SHIFTED_BITS = SIGNED_MANT_BITS + 3  // 8
    , parameter LOG2_SUB_GROUP_SIZE = $clog2(sub_group_size)  // 3
    , parameter SUB_ACC_BITS = SHIFTED_BITS + LOG2_SUB_GROUP_SIZE  // 11
    , parameter INVMX_FULL_MANT_BITS = invmx_mant_bits + 1  // 4
    , parameter SUB_MULT_BITS = SUB_ACC_BITS + INVMX_FULL_MANT_BITS  // 15
    , parameter LOG2_SUBS_PER_WGT = $clog2(SUBS_PER_WGT)  // 1
    , parameter MULT_OUT_BITS = SUB_MULT_BITS + LOG2_SUBS_PER_WGT  // 16
    , parameter TOTAL_FIXED_SCALE = 3 + invmx_mant_bits  // 6
) (
    input logic clk
    , input logic rst_n

    // MXINT4 weight: 2 groups (2 shared exponents + 32 mantissas)
    , input logic [NUM_WGT_GROUPS-1:0][mxint_exp_bits-1:0] mxint_exp
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant

    // INVMX activation: 1 group (max_base_exp + 32 signs/exps + 4 sub-group mantissas)
    , input logic [max_base_exp_bits-1:0] max_base_exp
    , input logic [group_size-1:0] invmx_sign
    , input logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp
    , input logic [NUM_SUB_GROUPS-1:0][invmx_mant_bits-1:0] invmx_mant

    // Output: one (exp, sum) per weight group
    , output logic [NUM_WGT_GROUPS-1:0][mxint_exp_bits:0] out_exp
    , output logic signed [NUM_WGT_GROUPS-1:0][MULT_OUT_BITS-1:0] out_sum
);

    // ----------------------------------------------------------------
    // Step 1: Apply INVMX sign to MXINT mantissa (zero if invmx_exp==7)
    // ----------------------------------------------------------------
    logic signed [group_size-1:0][SIGNED_MANT_BITS-1:0] signed_mant;

    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            if (invmx_exp[i] == 3'b111)
                signed_mant[i] = '0;
            else if (invmx_sign[i])
                signed_mant[i] = -SIGNED_MANT_BITS'(signed'(mxint_mant[i]));
            else
                signed_mant[i] =  SIGNED_MANT_BITS'(signed'(mxint_mant[i]));
        end
    end

    // ----------------------------------------------------------------
    // Step 2: Append 3 zeros, right-shift by invmx_exp
    // ----------------------------------------------------------------
    logic signed [group_size-1:0][SHIFTED_BITS-1:0] shifted_mant;

    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            shifted_mant[i] = $signed({signed_mant[i], {3{1'b0}}}) >>> invmx_exp[i];
        end
    end

    // ----------------------------------------------------------------
    // Phase 1: 4 sub-group adder trees (8 elements each)
    // ----------------------------------------------------------------
    genvar sg, s, j;
    generate
        for (sg = 0; sg < NUM_SUB_GROUPS; sg++) begin : sub_grp
            for (s = 0; s <= LOG2_SUB_GROUP_SIZE; s++) begin : tree
                localparam int W = SHIFTED_BITS + s;
                localparam int N = sub_group_size >> s;
                logic signed [W-1:0] val [N-1:0];

                if (s == 0) begin : init
                    for (j = 0; j < N; j++) begin : elem
                        assign val[j] = shifted_mant[sg * sub_group_size + j];
                    end
                end else begin : reduce
                    for (j = 0; j < N; j++) begin : elem
                        assign val[j] = W'(signed'(tree[s-1].val[2*j]))
                                      + W'(signed'(tree[s-1].val[2*j+1]));
                    end
                end
            end
        end
    endgenerate

    // ----------------------------------------------------------------
    // Phase 2: Multiply each sub-group sum by its shared mantissa {1, mmm}
    // ----------------------------------------------------------------
    logic signed [NUM_SUB_GROUPS-1:0][SUB_MULT_BITS-1:0] sub_mult_result;

    always_comb begin
        sub_mult_result[0] = SUB_MULT_BITS'(signed'(sub_grp[0].tree[LOG2_SUB_GROUP_SIZE].val[0]))
                           * SUB_MULT_BITS'(signed'({1'b0, 1'b1, invmx_mant[0]}));
        sub_mult_result[1] = SUB_MULT_BITS'(signed'(sub_grp[1].tree[LOG2_SUB_GROUP_SIZE].val[0]))
                           * SUB_MULT_BITS'(signed'({1'b0, 1'b1, invmx_mant[1]}));
        sub_mult_result[2] = SUB_MULT_BITS'(signed'(sub_grp[2].tree[LOG2_SUB_GROUP_SIZE].val[0]))
                           * SUB_MULT_BITS'(signed'({1'b0, 1'b1, invmx_mant[2]}));
        sub_mult_result[3] = SUB_MULT_BITS'(signed'(sub_grp[3].tree[LOG2_SUB_GROUP_SIZE].val[0]))
                           * SUB_MULT_BITS'(signed'({1'b0, 1'b1, invmx_mant[3]}));
    end

    // ----------------------------------------------------------------
    // Phase 3: Sum sub-group products by weight group
    //   Weight group 0 (elements 0-15):  sub_grp 0 + sub_grp 1
    //   Weight group 1 (elements 16-31): sub_grp 2 + sub_grp 3
    // ----------------------------------------------------------------
    logic signed [NUM_WGT_GROUPS-1:0][MULT_OUT_BITS-1:0] wgt_sum;
    logic [NUM_WGT_GROUPS-1:0][mxint_exp_bits:0] combined_exp_pre;

    always_comb begin
        wgt_sum[0] = MULT_OUT_BITS'(signed'(sub_mult_result[0]))
                   + MULT_OUT_BITS'(signed'(sub_mult_result[1]));
        wgt_sum[1] = MULT_OUT_BITS'(signed'(sub_mult_result[2]))
                   + MULT_OUT_BITS'(signed'(sub_mult_result[3]));

        combined_exp_pre[0] = (mxint_exp_bits + 1)'(mxint_exp[0])
                            + (mxint_exp_bits + 1)'(max_base_exp)
                            - (mxint_exp_bits + 1)'(TOTAL_FIXED_SCALE);
        combined_exp_pre[1] = (mxint_exp_bits + 1)'(mxint_exp[1])
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
            out_sum <= wgt_sum;
        end
    end

endmodule
