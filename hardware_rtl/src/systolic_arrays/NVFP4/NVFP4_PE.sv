// Block-32 NVFP4 PE: two 16-element NVFP4 sub-blocks per PE (each with its own
// E4M3 scale). Each sub-block is scaled and normalized to FP32, then the two
// sub-block values are summed with the running accumulator in ONE fused 3-input
// FP add (the normalize + accumulate are fused, same backend style as MXFP4_PE /
// MiX45b_MXINT4_fused — no register between normalize and accumulate).
module NVFP4_PE #(
    parameter group_size = 32
    , parameter sub_group_size = 16
    , parameter scale_bits = 8
    , parameter elem_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [group_size/sub_group_size-1:0][scale_bits-1:0] nvfp4_a_scale_in
    , input logic [group_size-1:0][elem_bits-1:0] nvfp4_a_elem_in
    , input logic [group_size/sub_group_size-1:0][scale_bits-1:0] nvfp4_b_scale_in
    , input logic [group_size-1:0][elem_bits-1:0] nvfp4_b_elem_in

    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    , output logic [group_size/sub_group_size-1:0][scale_bits-1:0] nvfp4_a_scale_out
    , output logic [group_size-1:0][elem_bits-1:0] nvfp4_a_elem_out
    , output logic [group_size/sub_group_size-1:0][scale_bits-1:0] nvfp4_b_scale_out
    , output logic [group_size-1:0][elem_bits-1:0] nvfp4_b_elem_out

    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    localparam PRODUCT_WIDTH = 9;
    localparam FIXED_SCALE = 2;
    localparam SCALE_EXP_BITS = 4;
    localparam SCALE_MANT_BITS = 3;
    localparam SCALE_MANT_PRODUCT_BITS = 8;
    localparam NUM_SUB = group_size / sub_group_size;             // 2
    localparam LOG2_SUB_SIZE = $clog2(sub_group_size);            // 4
    localparam DOT_OUT_BITS = PRODUCT_WIDTH + LOG2_SUB_SIZE;      // 13
    localparam SCALE_EXP_SUM_BITS = SCALE_EXP_BITS + 1;           // 5
    localparam TOTAL_FIXED_SCALE = FIXED_SCALE + 2 * SCALE_MANT_BITS; // 8
    localparam EXT_PRODUCT_BITS = DOT_OUT_BITS + SCALE_MANT_PRODUCT_BITS; // 21

    // MUL outputs (one per sub-block)
    logic [NUM_SUB-1:0][SCALE_EXP_SUM_BITS-1:0] mul_scale_exp_sum;
    logic signed [NUM_SUB-1:0][DOT_OUT_BITS-1:0] mul_out_sum;
    logic [NUM_SUB-1:0][SCALE_MANT_PRODUCT_BITS-1:0] mul_scale_mant_product;

    // Per-sub-block normalize (combinational) → FP32
    logic [NUM_SUB-1:0] norm_sign_pre;
    logic [NUM_SUB-1:0][fp_exp_bits-1:0] norm_exp_pre;
    logic [NUM_SUB-1:0][fp_mant_bits-1:0] norm_mant_pre;
    logic [DOT_OUT_BITS-1:0] mul_sum_abs [NUM_SUB];
    logic [EXT_PRODUCT_BITS-1:0] extended_product [NUM_SUB];
    logic [fp_mant_bits:0] ext_mant_full [NUM_SUB];
    logic [fp_exp_bits:0] raw_exp [NUM_SUB];
    logic [fp_exp_bits:0] norm_exp_ext [NUM_SUB];
    int ext_lead_idx [NUM_SUB];

    // Local FP32 accumulator
    logic local_acc_sign;
    logic [fp_exp_bits-1:0] local_acc_exp;
    logic [fp_mant_bits-1:0] local_acc_mant;

    // 3-input FP32 accumulate signals (norm[0], norm[1], local_acc)
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

    // ----------------------------------------------------------------
    // Stage 1: NVFP4 block-32 multiply (2 sub-blocks, registered)
    // ----------------------------------------------------------------
    NVFP4_MUL #(
        .group_size(group_size)
        , .sub_group_size(sub_group_size)
        , .scale_bits(scale_bits)
        , .elem_bits(elem_bits)
    ) u_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .nvfp4_a_scale(nvfp4_a_scale_in)
        , .nvfp4_a_elem(nvfp4_a_elem_in)
        , .nvfp4_b_scale(nvfp4_b_scale_in)
        , .nvfp4_b_elem(nvfp4_b_elem_in)
        , .out_scale_exp_sum(mul_scale_exp_sum)
        , .out_sum(mul_out_sum)
        , .out_scale_mant_product(mul_scale_mant_product)
    );

    // ----------------------------------------------------------------
    // Stage 2a: scale + normalize each sub-block to FP32 (combinational).
    //   Identical per-sub-block math to NVFP4_PE; fused with the 3-input
    //   accumulate below.
    // ----------------------------------------------------------------
    always_comb begin
        for (int g = 0; g < NUM_SUB; g++) begin
            norm_sign_pre[g] = mul_out_sum[g][DOT_OUT_BITS-1];
            mul_sum_abs[g] = norm_sign_pre[g]
                ? DOT_OUT_BITS'(~mul_out_sum[g] + 1'b1)
                : DOT_OUT_BITS'(mul_out_sum[g]);

            extended_product[g] = mul_sum_abs[g] * mul_scale_mant_product[g];

            ext_lead_idx[g] = -1;
            ext_mant_full[g] = '0;
            raw_exp[g] = '0;
            norm_exp_ext[g] = '0;
            norm_exp_pre[g] = '0;
            norm_mant_pre[g] = '0;

            for (int i = EXT_PRODUCT_BITS - 1; i >= 0; i--) begin
                if ((ext_lead_idx[g] == -1) && extended_product[g][i]) begin
                    ext_lead_idx[g] = i;
                end
            end

            if (ext_lead_idx[g] >= 0) begin
                // EXT_PRODUCT_BITS (21) <= fp_mant_bits (23): always left shift
                ext_mant_full[g] = (fp_mant_bits + 1)'(extended_product[g]) << (fp_mant_bits - ext_lead_idx[g]);

                raw_exp[g] = (fp_exp_bits + 1)'(mul_scale_exp_sum[g])
                           + (fp_exp_bits + 1)'(ext_lead_idx[g]);

                if (raw_exp[g] >= (fp_exp_bits + 1)'(TOTAL_FIXED_SCALE)) begin
                    norm_exp_ext[g] = raw_exp[g] - (fp_exp_bits + 1)'(TOTAL_FIXED_SCALE);
                    if (norm_exp_ext[g][fp_exp_bits]) begin
                        norm_exp_pre[g]  = {fp_exp_bits{1'b1}};
                        norm_mant_pre[g] = {fp_mant_bits{1'b1}};
                    end else begin
                        norm_exp_pre[g]  = norm_exp_ext[g][fp_exp_bits-1:0];
                        norm_mant_pre[g] = ext_mant_full[g][fp_mant_bits-1:0];
                    end
                end
                // else underflow stays at zero
            end
        end
    end

    // ----------------------------------------------------------------
    // Stage 2b: 3-input FP32 accumulate (fused with 2a, registered into acc)
    //   Inputs: norm[0], norm[1], local_acc
    // ----------------------------------------------------------------
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

    // ----------------------------------------------------------------
    // Systolic pass-through: activation flows right, weight flows down
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvfp4_a_scale_out <= '0;
            nvfp4_a_elem_out  <= '0;
        end else begin
            nvfp4_a_scale_out <= nvfp4_a_scale_in;
            nvfp4_a_elem_out  <= nvfp4_a_elem_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvfp4_b_scale_out <= '0;
            nvfp4_b_elem_out  <= '0;
        end else begin
            nvfp4_b_scale_out <= nvfp4_b_scale_in;
            nvfp4_b_elem_out  <= nvfp4_b_elem_in;
        end
    end

    // ----------------------------------------------------------------
    // Accumulator output and local register
    // ----------------------------------------------------------------
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
