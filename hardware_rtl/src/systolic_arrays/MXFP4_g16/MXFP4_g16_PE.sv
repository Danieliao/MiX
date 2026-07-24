// Block-32-PE MXFP4 (4.5b): two 16-element sub-blocks (each its own E8M0 scale),
// each normalized to FP32, then summed with the running accumulator in ONE fused
// 3-input FP add (normalize + accumulate fused, MXFP4-style backend).
module MXFP4_g16_PE #(
    parameter group_size = 32
    , parameter sub_group_size = 16
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [group_size/sub_group_size-1:0][shared_exp_bits-1:0] mxfp4_a_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem_in
    , input logic [group_size/sub_group_size-1:0][shared_exp_bits-1:0] mxfp4_b_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem_in

    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    , output logic [group_size/sub_group_size-1:0][shared_exp_bits-1:0] mxfp4_a_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem_out
    , output logic [group_size/sub_group_size-1:0][shared_exp_bits-1:0] mxfp4_b_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem_out

    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    localparam PRODUCT_WIDTH = 9;
    localparam NUM_SUB = group_size / sub_group_size;            // 2
    localparam LOG2_SUB_SIZE = $clog2(sub_group_size);           // 4
    localparam DOT_OUT_BITS = PRODUCT_WIDTH + LOG2_SUB_SIZE;     // 13
    localparam MUL_EXP_BITS = shared_exp_bits + 1;               // 9

    logic [NUM_SUB-1:0][MUL_EXP_BITS-1:0] mul_out_exp;
    logic signed [NUM_SUB-1:0][DOT_OUT_BITS-1:0] mul_out_sum;

    // Per-sub-block normalize → FP32 (combinational)
    logic [NUM_SUB-1:0] norm_sign_pre;
    logic [NUM_SUB-1:0][fp_exp_bits-1:0] norm_exp_pre;
    logic [NUM_SUB-1:0][fp_mant_bits-1:0] norm_mant_pre;
    logic [DOT_OUT_BITS-1:0] mul_sum_abs [NUM_SUB];
    logic [fp_mant_bits:0] mul_mant_full [NUM_SUB];
    logic [fp_exp_bits:0] mul_exp_ext [NUM_SUB];
    int mul_lead_idx [NUM_SUB];

    logic local_acc_sign;
    logic [fp_exp_bits-1:0] local_acc_exp;
    logic [fp_mant_bits-1:0] local_acc_mant;

    // 3-input FP32 accumulate signals
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

    MXFP4_g16_MUL #(
        .group_size(group_size)
        , .sub_group_size(sub_group_size)
        , .shared_exp_bits(shared_exp_bits)
        , .elem_bits(elem_bits)
    ) u_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .mxfp4_a_exp(mxfp4_a_exp_in)
        , .mxfp4_a_elem(mxfp4_a_elem_in)
        , .mxfp4_b_exp(mxfp4_b_exp_in)
        , .mxfp4_b_elem(mxfp4_b_elem_in)
        , .out_exp(mul_out_exp)
        , .out_sum(mul_out_sum)
    );

    // Stage 2a: normalize each sub-block (integer-sum normalize; E8M0 has no mantissa)
    always_comb begin
        for (int g = 0; g < NUM_SUB; g++) begin
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

                mul_exp_ext[g] = (fp_exp_bits + 1)'(mul_out_exp[g]) + (fp_exp_bits + 1)'(mul_lead_idx[g]);
                if (mul_exp_ext[g][fp_exp_bits]) begin
                    norm_exp_pre[g]  = {fp_exp_bits{1'b1}};
                    norm_mant_pre[g] = {fp_mant_bits{1'b1}};
                end else begin
                    norm_exp_pre[g]  = mul_exp_ext[g][fp_exp_bits-1:0];
                    norm_mant_pre[g] = mul_mant_full[g][fp_mant_bits-1:0];
                end
            end
        end
    end

    // Stage 2b: 3-input FP32 accumulate (fused with 2a) — norm[0] + norm[1] + local_acc
    always_comb begin
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

        mant_a_signed = norm_sign_pre[0] ? -$signed({1'b0, mant_a_aligned}) : $signed({1'b0, mant_a_aligned});
        mant_b_signed = norm_sign_pre[1] ? -$signed({1'b0, mant_b_aligned}) : $signed({1'b0, mant_b_aligned});
        mant_c_signed = local_acc_sign   ? -$signed({1'b0, mant_c_aligned}) : $signed({1'b0, mant_c_aligned});

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

    // Systolic pass-through: activation flows right, weight flows down
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxfp4_a_exp_out  <= '0;
            mxfp4_a_elem_out <= '0;
        end else begin
            mxfp4_a_exp_out  <= mxfp4_a_exp_in;
            mxfp4_a_elem_out <= mxfp4_a_elem_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxfp4_b_exp_out  <= '0;
            mxfp4_b_elem_out <= '0;
        end else begin
            mxfp4_b_exp_out  <= mxfp4_b_exp_in;
            mxfp4_b_elem_out <= mxfp4_b_elem_in;
        end
    end

    assign acc_sign_out = local_acc_sign;
    assign acc_exp_out  = local_acc_exp;
    assign acc_mant_out = local_acc_mant;

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

endmodule
