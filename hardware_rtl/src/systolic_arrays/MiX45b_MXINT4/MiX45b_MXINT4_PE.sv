module MiX45b_MXINT4_PE #(
    parameter group_size = 32
    , parameter wgt_group_size = 16
    , parameter sub_group_size = 8
    , parameter mxint_exp_bits = 8
    , parameter mxint_mant_bits = 4
    , parameter max_base_exp_bits = 5
    , parameter invmx_exp_bits = 3
    , parameter invmx_mant_bits = 3
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    // MXINT4 weight (flows right): 2 groups
    , input logic [group_size/wgt_group_size-1:0][mxint_exp_bits-1:0] mxint_exp_in
    , input logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant_in

    // INVMX activation (flows down): 1 group with 4 sub-group mantissas
    , input logic [max_base_exp_bits-1:0] invmx_max_base_exp_in
    , input logic [group_size-1:0] invmx_sign_in
    , input logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_in
    , input logic [group_size/sub_group_size-1:0][invmx_mant_bits-1:0] invmx_mant_in

    // Accumulator input (from above)
    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    // MXINT4 weight pass-through (to right)
    , output logic [group_size/wgt_group_size-1:0][mxint_exp_bits-1:0] mxint_exp_out
    , output logic signed [group_size-1:0][mxint_mant_bits-1:0] mxint_mant_out

    // INVMX activation pass-through (to below)
    , output logic [max_base_exp_bits-1:0] invmx_max_base_exp_out
    , output logic [group_size-1:0] invmx_sign_out
    , output logic [group_size-1:0][invmx_exp_bits-1:0] invmx_exp_out
    , output logic [group_size/sub_group_size-1:0][invmx_mant_bits-1:0] invmx_mant_out

    // Accumulator output (to below)
    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    // Derived localparams matching MUL
    localparam NUM_WGT_GROUPS = group_size / wgt_group_size;
    localparam NUM_SUB_GROUPS = group_size / sub_group_size;
    localparam SUBS_PER_WGT = wgt_group_size / sub_group_size;
    localparam SIGNED_MANT_BITS = mxint_mant_bits;
    localparam SHIFTED_BITS = SIGNED_MANT_BITS + 3;
    localparam LOG2_SUB_GROUP_SIZE = $clog2(sub_group_size);
    localparam SUB_ACC_BITS = SHIFTED_BITS + LOG2_SUB_GROUP_SIZE;
    localparam INVMX_FULL_MANT_BITS = invmx_mant_bits + 1;
    localparam SUB_MULT_BITS = SUB_ACC_BITS + INVMX_FULL_MANT_BITS;
    localparam LOG2_SUBS_PER_WGT = $clog2(SUBS_PER_WGT);
    localparam DOT_OUT_BITS = SUB_MULT_BITS + LOG2_SUBS_PER_WGT;
    localparam MUL_EXP_BITS = mxint_exp_bits + 1;

    // ================================================================
    // Stage 1: MUL (registered inside MUL)
    //   Outputs 2 (exp, sum) pairs — one per weight group
    // ================================================================
    logic [NUM_WGT_GROUPS-1:0][MUL_EXP_BITS-1:0] mul_out_exp;
    logic signed [NUM_WGT_GROUPS-1:0][DOT_OUT_BITS-1:0] mul_out_sum;

    MiX45b_MXINT4_MUL #(
        .group_size(group_size)
        , .wgt_group_size(wgt_group_size)
        , .sub_group_size(sub_group_size)
        , .mxint_exp_bits(mxint_exp_bits)
        , .mxint_mant_bits(mxint_mant_bits)
        , .max_base_exp_bits(max_base_exp_bits)
        , .invmx_exp_bits(invmx_exp_bits)
        , .invmx_mant_bits(invmx_mant_bits)
    ) u_mix_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .mxint_exp(mxint_exp_in)
        , .mxint_mant(mxint_mant_in)
        , .max_base_exp(invmx_max_base_exp_in)
        , .invmx_sign(invmx_sign_in)
        , .invmx_exp(invmx_exp_in)
        , .invmx_mant(invmx_mant_in)
        , .out_exp(mul_out_exp)
        , .out_sum(mul_out_sum)
    );

    // ================================================================
    // Stage 2a: Parallel normalize (2 weight groups → 2 FP32 values, combinational)
    //   FUSED with the 3-input accumulate below (no register between them), so the
    //   shared FP32 backend cells are sized like the MXFP4 baseline (fair compare).
    // ================================================================
    logic [1:0][DOT_OUT_BITS-1:0] mul_sum_abs;
    logic [1:0][fp_mant_bits:0] mul_mant_full;
    logic [1:0][fp_exp_bits:0] mul_exp_ext;
    int mul_lead_idx [2];

    logic [1:0] norm_sign_pre;
    logic [1:0][fp_exp_bits-1:0] norm_exp_pre;
    logic [1:0][fp_mant_bits-1:0] norm_mant_pre;

    always_comb begin
        for (int g = 0; g < 2; g++) begin
            norm_sign_pre[g] = mul_out_sum[g][DOT_OUT_BITS-1];
            mul_sum_abs[g] = norm_sign_pre[g]
                ? DOT_OUT_BITS'(~mul_out_sum[g] + 1'b1)
                : DOT_OUT_BITS'(mul_out_sum[g]);
            mul_lead_idx[g] = -1;
            mul_mant_full[g] = '0;
            mul_exp_ext[g] = '0;
            norm_exp_pre[g] = '0;
            norm_mant_pre[g] = '0;

            for (int i = DOT_OUT_BITS - 1; i >= 0; i--) begin
                if ((mul_lead_idx[g] == -1) && mul_sum_abs[g][i]) begin
                    mul_lead_idx[g] = i;
                end
            end

            if (mul_lead_idx[g] >= 0) begin
                if (mul_lead_idx[g] > fp_mant_bits) begin
                    mul_mant_full[g] = {1'b0, mul_sum_abs[g]} >> (mul_lead_idx[g] - fp_mant_bits);
                end else begin
                    mul_mant_full[g] = (fp_mant_bits + 1)'(mul_sum_abs[g]) << (fp_mant_bits - mul_lead_idx[g]);
                end

                mul_exp_ext[g] = (fp_exp_bits + 1)'(mul_out_exp[g])
                               + (fp_exp_bits + 1)'(mul_lead_idx[g]);
                if (mul_exp_ext[g][fp_exp_bits]) begin
                    norm_exp_pre[g] = {fp_exp_bits{1'b1}};
                    norm_mant_pre[g] = {fp_mant_bits{1'b1}};
                end else begin
                    norm_exp_pre[g] = mul_exp_ext[g][fp_exp_bits-1:0];
                    norm_mant_pre[g] = mul_mant_full[g][fp_mant_bits-1:0];
                end
            end
        end
    end

    // ================================================================
    // Stage 2b: 3-input FP32 accumulate (fused with 2a, registered into the accumulator)
    //   Inputs: norm[0], norm[1], local_acc
    // ================================================================
    logic local_acc_sign;
    logic [fp_exp_bits-1:0] local_acc_exp;
    logic [fp_mant_bits-1:0] local_acc_mant;

    logic [fp_exp_bits-1:0] max3_exp;
    logic [fp_exp_bits-1:0] diff_a, diff_b, diff_c;
    logic [fp_mant_bits:0] mant_a_ext, mant_b_ext, mant_c_ext;
    logic [fp_mant_bits:0] mant_a_aligned, mant_b_aligned, mant_c_aligned;
    logic signed [fp_mant_bits+1:0] mant_a_signed, mant_b_signed, mant_c_signed;
    logic signed [fp_mant_bits+3:0] sum3_mant;
    logic [fp_mant_bits+2:0] sum3_mant_abs;
    logic acc3_sign_pre;
    logic [fp_exp_bits-1:0] acc3_exp_pre;
    logic [fp_mant_bits-1:0] acc3_mant_pre;
    logic [fp_mant_bits:0] acc3_normalized_mant;
    logic [fp_exp_bits:0] acc3_normalized_exp;
    int acc3_lead_idx;

    always_comb begin
        // Find max exponent among three
        if (norm_exp_pre[0] >= norm_exp_pre[1]) begin
            max3_exp = (norm_exp_pre[0] >= local_acc_exp) ? norm_exp_pre[0] : local_acc_exp;
        end else begin
            max3_exp = (norm_exp_pre[1] >= local_acc_exp) ? norm_exp_pre[1] : local_acc_exp;
        end

        diff_a = max3_exp - norm_exp_pre[0];
        diff_b = max3_exp - norm_exp_pre[1];
        diff_c = max3_exp - local_acc_exp;

        mant_a_ext = ((norm_exp_pre[0] == '0) && (norm_mant_pre[0] == '0)) ? '0 : {1'b1, norm_mant_pre[0]};
        mant_b_ext = ((norm_exp_pre[1] == '0) && (norm_mant_pre[1] == '0)) ? '0 : {1'b1, norm_mant_pre[1]};
        mant_c_ext = ((local_acc_exp == '0) && (local_acc_mant == '0)) ? '0 : {1'b1, local_acc_mant};

        mant_a_aligned = mant_a_ext >> diff_a;
        mant_b_aligned = mant_b_ext >> diff_b;
        mant_c_aligned = mant_c_ext >> diff_c;

        mant_a_signed = norm_sign_pre[0]    ? -$signed({1'b0, mant_a_aligned}) : $signed({1'b0, mant_a_aligned});
        mant_b_signed = norm_sign_pre[1]    ? -$signed({1'b0, mant_b_aligned}) : $signed({1'b0, mant_b_aligned});
        mant_c_signed = local_acc_sign  ? -$signed({1'b0, mant_c_aligned}) : $signed({1'b0, mant_c_aligned});

        sum3_mant = (fp_mant_bits + 4)'(signed'(mant_a_signed))
                  + (fp_mant_bits + 4)'(signed'(mant_b_signed))
                  + (fp_mant_bits + 4)'(signed'(mant_c_signed));

        if (sum3_mant[fp_mant_bits+3]) begin
            acc3_sign_pre = 1'b1;
            sum3_mant_abs = (fp_mant_bits + 3)'(-sum3_mant[fp_mant_bits+2:0]);
        end else begin
            acc3_sign_pre = 1'b0;
            sum3_mant_abs = sum3_mant[fp_mant_bits+2:0];
        end

        acc3_lead_idx = -1;
        acc3_normalized_mant = '0;
        acc3_normalized_exp = '0;
        acc3_exp_pre = '0;
        acc3_mant_pre = '0;

        for (int i = fp_mant_bits + 2; i >= 0; i--) begin
            if ((acc3_lead_idx == -1) && sum3_mant_abs[i]) begin
                acc3_lead_idx = i;
            end
        end

        if (acc3_lead_idx >= 0) begin
            if (acc3_lead_idx > fp_mant_bits) begin
                acc3_normalized_mant = {1'b0, sum3_mant_abs} >> (acc3_lead_idx - fp_mant_bits);
            end else begin
                acc3_normalized_mant = (fp_mant_bits + 1)'(sum3_mant_abs) << (fp_mant_bits - acc3_lead_idx);
            end

            if (acc3_lead_idx > fp_mant_bits) begin
                acc3_normalized_exp = (fp_exp_bits + 1)'(max3_exp) + (acc3_lead_idx - fp_mant_bits);
            end else begin
                if (max3_exp >= (fp_mant_bits - acc3_lead_idx)) begin
                    acc3_normalized_exp = (fp_exp_bits + 1)'(max3_exp) - (fp_mant_bits - acc3_lead_idx);
                end else begin
                    acc3_normalized_exp = '0;
                end
            end

            if (acc3_normalized_exp[fp_exp_bits]) begin
                acc3_exp_pre = {fp_exp_bits{1'b1}};
                acc3_mant_pre = {fp_mant_bits{1'b1}};
            end else begin
                acc3_exp_pre = acc3_normalized_exp[fp_exp_bits-1:0];
                acc3_mant_pre = acc3_normalized_mant[fp_mant_bits-1:0];
            end
        end else begin
            acc3_sign_pre = 1'b0;
        end
    end

    // Stage 3 register (local accumulator)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_acc_sign <= '0;
            local_acc_exp  <= '0;
            local_acc_mant <= '0;
        end else if (acc_shift) begin
            local_acc_sign <= acc_sign_in;
            local_acc_exp  <= acc_exp_in;
            local_acc_mant <= acc_mant_in;
        end else begin
            local_acc_sign <= acc3_sign_pre;
            local_acc_exp  <= acc3_exp_pre;
            local_acc_mant <= acc3_mant_pre;
        end
    end

    // ================================================================
    // Pass-through registers (1-cycle latency)
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxint_exp_out  <= '0;
            mxint_mant_out <= '0;
        end else begin
            mxint_exp_out  <= mxint_exp_in;
            mxint_mant_out <= mxint_mant_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            invmx_max_base_exp_out <= '0;
            invmx_sign_out         <= '0;
            invmx_exp_out          <= '0;
            invmx_mant_out         <= '0;
        end else begin
            invmx_max_base_exp_out <= invmx_max_base_exp_in;
            invmx_sign_out         <= invmx_sign_in;
            invmx_exp_out          <= invmx_exp_in;
            invmx_mant_out         <= invmx_mant_in;
        end
    end

    assign acc_sign_out = local_acc_sign;
    assign acc_exp_out  = local_acc_exp;
    assign acc_mant_out = local_acc_mant;

endmodule
