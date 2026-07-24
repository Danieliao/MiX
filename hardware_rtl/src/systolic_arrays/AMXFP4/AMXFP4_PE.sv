module AMXFP4_PE #(
    parameter group_size = 32
    , parameter scale_bits = 7
    , parameter elem_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [scale_bits-1:0] amxfp4_a_scale_pos_in
    , input logic [scale_bits-1:0] amxfp4_a_scale_neg_in
    , input logic [group_size-1:0][elem_bits-1:0] amxfp4_a_elem_in
    , input logic [scale_bits-1:0] amxfp4_b_scale_pos_in
    , input logic [scale_bits-1:0] amxfp4_b_scale_neg_in
    , input logic [group_size-1:0][elem_bits-1:0] amxfp4_b_elem_in

    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    , output logic [scale_bits-1:0] amxfp4_a_scale_pos_out
    , output logic [scale_bits-1:0] amxfp4_a_scale_neg_out
    , output logic [group_size-1:0][elem_bits-1:0] amxfp4_a_elem_out
    , output logic [scale_bits-1:0] amxfp4_b_scale_pos_out
    , output logic [scale_bits-1:0] amxfp4_b_scale_neg_out
    , output logic [group_size-1:0][elem_bits-1:0] amxfp4_b_elem_out

    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    // Derived constants
    localparam PRODUCT_WIDTH = 9;
    localparam FIXED_SCALE = 2;
    localparam SCALE_EXP_BITS = 5;
    localparam SCALE_MANT_BITS = 2;
    localparam SCALE_FULL_MANT_W = SCALE_MANT_BITS + 1;             // 3
    localparam SCALE_MANT_PRODUCT_BITS = 2 * SCALE_FULL_MANT_W;     // 6
    localparam LOG2_GROUP_SIZE = $clog2(group_size);
    localparam DOT_OUT_BITS = PRODUCT_WIDTH + LOG2_GROUP_SIZE;      // 14
    localparam SCALE_EXP_SUM_BITS = SCALE_EXP_BITS + 1;            // 6
    localparam TOTAL_FIXED_SCALE = FIXED_SCALE + 2 * SCALE_MANT_BITS; // 6
    localparam EXT_PRODUCT_BITS = DOT_OUT_BITS + SCALE_MANT_PRODUCT_BITS; // 20
    localparam COMBINE_BITS = EXT_PRODUCT_BITS + 3;                // sign + 4-input growth = 23

    // MUL outputs (four quadrants {PP,PN,NP,NN})
    logic [3:0][SCALE_EXP_SUM_BITS-1:0] mul_scale_exp_sum;
    logic signed [3:0][DOT_OUT_BITS-1:0] mul_out_sum;
    logic [3:0][SCALE_MANT_PRODUCT_BITS-1:0] mul_scale_mant_product;

    // Per-quadrant scale-apply signals
    logic signed [DOT_OUT_BITS-1:0]      sum_q       [3:0];
    logic                                q_sign      [3:0];
    logic [DOT_OUT_BITS-1:0]             sum_abs_q   [3:0];
    logic [EXT_PRODUCT_BITS-1:0]         ext_abs_q   [3:0];
    logic [SCALE_EXP_SUM_BITS-1:0]       exp_q       [3:0];
    logic [SCALE_EXP_SUM_BITS-1:0]       exp_diff_q  [3:0];
    logic [EXT_PRODUCT_BITS-1:0]         aligned_q   [3:0];
    logic signed [COMBINE_BITS-1:0]      term_q      [3:0];
    logic [SCALE_EXP_SUM_BITS-1:0]       comb_max_exp;
    logic signed [COMBINE_BITS-1:0]      comb_sum;
    logic                                comb_sign;
    logic [COMBINE_BITS-2:0]             comb_abs;

    // Combined-to-FP32 normalization signals (Stage 2 combinational)
    logic block_sign_pre;
    logic [fp_exp_bits-1:0] block_exp_pre;
    logic [fp_mant_bits-1:0] block_mant_pre;
    int   comb_lead_idx;
    logic [fp_mant_bits:0] comb_mant_full;
    logic [fp_exp_bits:0] raw_exp;
    logic [fp_exp_bits:0] block_exp_ext;

    // Stage 2 pipeline register: the combined fixed-point dot product (before
    // normalize). Only the feed-forward 4-quadrant combine is split out into its
    // own stage; the normalize + FP32 accumulate are FUSED in the final stage,
    // identical to the MXFP4 backend, so the shared FP32 cells are sized the same
    // way as the MXFP4 baseline (fair area/power comparison).
    logic signed [COMBINE_BITS-1:0] comb_sum_q;
    logic [SCALE_EXP_SUM_BITS-1:0]  comb_max_exp_q;

    // Local FP32 accumulator
    logic local_acc_sign;
    logic [fp_exp_bits-1:0] local_acc_exp;
    logic [fp_mant_bits-1:0] local_acc_mant;

    // FP32 addition signals
    logic [fp_exp_bits-1:0] exp_diff;
    logic acc_exp_larger;
    logic [fp_exp_bits-1:0] max_exp;
    logic [fp_mant_bits:0] acc_mant_ext;
    logic [fp_mant_bits:0] mul_mant_ext;
    logic [fp_mant_bits:0] acc_mant_aligned;
    logic [fp_mant_bits:0] mul_mant_aligned;
    logic signed [fp_mant_bits+1:0] acc_mant_signed;
    logic signed [fp_mant_bits+1:0] mul_mant_signed;
    logic signed [fp_mant_bits+2:0] sum_mant;
    logic [fp_mant_bits+1:0] sum_mant_abs;
    logic acc_sign_pre;
    logic [fp_exp_bits-1:0] acc_exp_pre;
    logic [fp_mant_bits-1:0] acc_mant_pre;
    logic [fp_mant_bits:0] normalized_mant;
    logic [fp_exp_bits:0] normalized_exp;
    int add_lead_idx;

    // ----------------------------------------------------------------
    // Stage 1: AMXFP4 multiply (E2M1 decode + 4 quadrant adder trees, registered)
    // ----------------------------------------------------------------
    AMXFP4_MUL #(
        .group_size(group_size)
        , .scale_bits(scale_bits)
        , .elem_bits(elem_bits)
    ) u_amxfp4_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .amxfp4_a_scale_pos(amxfp4_a_scale_pos_in)
        , .amxfp4_a_scale_neg(amxfp4_a_scale_neg_in)
        , .amxfp4_a_elem(amxfp4_a_elem_in)
        , .amxfp4_b_scale_pos(amxfp4_b_scale_pos_in)
        , .amxfp4_b_scale_neg(amxfp4_b_scale_neg_in)
        , .amxfp4_b_elem(amxfp4_b_elem_in)
        , .out_scale_exp_sum(mul_scale_exp_sum)
        , .out_sum(mul_out_sum)
        , .out_scale_mant_product(mul_scale_mant_product)
    );

    // ----------------------------------------------------------------
    // Stage 2a: apply the four result scales and combine on a shared exponent
    // ----------------------------------------------------------------
    always_comb begin
        // Per-quadrant magnitude product and dominant exponent
        comb_max_exp = '0;
        for (int q = 0; q < 4; q++) begin
            sum_q[q]     = $signed(mul_out_sum[q]);
            q_sign[q]    = sum_q[q][DOT_OUT_BITS-1];
            sum_abs_q[q] = q_sign[q] ? DOT_OUT_BITS'(-sum_q[q]) : DOT_OUT_BITS'(sum_q[q]);
            ext_abs_q[q] = sum_abs_q[q] * mul_scale_mant_product[q];
            exp_q[q]     = mul_scale_exp_sum[q];
            // Only quadrants with a non-zero contribution define the alignment exponent
            if ((ext_abs_q[q] != '0) && (exp_q[q] > comb_max_exp)) begin
                comb_max_exp = exp_q[q];
            end
        end

        // Align every quadrant down to comb_max_exp, re-sign, and accumulate
        comb_sum = '0;
        for (int q = 0; q < 4; q++) begin
            exp_diff_q[q] = comb_max_exp - exp_q[q];
            if (exp_diff_q[q] >= EXT_PRODUCT_BITS) begin
                aligned_q[q] = '0;
            end else begin
                aligned_q[q] = ext_abs_q[q] >> exp_diff_q[q];
            end
            term_q[q] = q_sign[q] ? -COMBINE_BITS'(signed'({1'b0, aligned_q[q]}))
                                  :  COMBINE_BITS'(signed'({1'b0, aligned_q[q]}));
            comb_sum  = comb_sum + term_q[q];
        end

    end

    // ----------------------------------------------------------------
    // Stage 2 register: latch the combined fixed-point sum + alignment exponent
    // (only the feed-forward combine is pipelined here)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            comb_sum_q     <= '0;
            comb_max_exp_q <= '0;
        end else begin
            comb_sum_q     <= comb_sum;
            comb_max_exp_q <= comb_max_exp;
        end
    end

    // ----------------------------------------------------------------
    // Stage 3a: normalize the combined fixed-point value to FP32.
    // Fused (same cycle) with the FP32 accumulate below, matching the MXFP4
    // backend so the shared FP32 cells are sized like the MXFP4 baseline.
    // ----------------------------------------------------------------
    always_comb begin
        comb_sign = comb_sum_q[COMBINE_BITS-1];
        comb_abs  = comb_sign ? (COMBINE_BITS-1)'(-comb_sum_q) : (COMBINE_BITS-1)'(comb_sum_q);

        comb_lead_idx  = -1;
        comb_mant_full = '0;
        raw_exp        = '0;
        block_exp_ext  = '0;
        block_sign_pre = comb_sign;
        block_exp_pre  = '0;
        block_mant_pre = '0;

        for (int i = COMBINE_BITS - 2; i >= 0; i--) begin
            if ((comb_lead_idx == -1) && comb_abs[i]) begin
                comb_lead_idx = i;
            end
        end

        if (comb_lead_idx >= 0) begin
            if (comb_lead_idx > fp_mant_bits) begin
                comb_mant_full = {1'b0, comb_abs} >> (comb_lead_idx - fp_mant_bits);
            end else begin
                comb_mant_full = (fp_mant_bits + 1)'(comb_abs) << (fp_mant_bits - comb_lead_idx);
            end

            raw_exp = (fp_exp_bits + 1)'(comb_max_exp_q) + (fp_exp_bits + 1)'(comb_lead_idx);

            if (raw_exp >= (fp_exp_bits + 1)'(TOTAL_FIXED_SCALE)) begin
                block_exp_ext = raw_exp - (fp_exp_bits + 1)'(TOTAL_FIXED_SCALE);
                if (block_exp_ext[fp_exp_bits]) begin
                    block_exp_pre  = {fp_exp_bits{1'b1}};
                    block_mant_pre = {fp_mant_bits{1'b1}};
                end else begin
                    block_exp_pre  = block_exp_ext[fp_exp_bits-1:0];
                    block_mant_pre = comb_mant_full[fp_mant_bits-1:0];
                end
            end
            // else: underflow stays at zero
        end
    end

    // ----------------------------------------------------------------
    // Stage 3b: FP32 accumulation (identical to MXFP4_PE), fused with 3a
    // ----------------------------------------------------------------
    always_comb begin
        if (local_acc_exp >= block_exp_pre) begin
            exp_diff = local_acc_exp - block_exp_pre;
            acc_exp_larger = 1'b1;
            max_exp = local_acc_exp;
        end else begin
            exp_diff = block_exp_pre - local_acc_exp;
            acc_exp_larger = 1'b0;
            max_exp = block_exp_pre;
        end

        acc_mant_ext = ((local_acc_exp == '0) && (local_acc_mant == '0)) ? '0 : {1'b1, local_acc_mant};
        mul_mant_ext = ((block_exp_pre == '0) && (block_mant_pre == '0)) ? '0 : {1'b1, block_mant_pre};

        if (acc_exp_larger) begin
            acc_mant_aligned = acc_mant_ext;
            mul_mant_aligned = mul_mant_ext >> exp_diff;
        end else begin
            acc_mant_aligned = acc_mant_ext >> exp_diff;
            mul_mant_aligned = mul_mant_ext;
        end

        acc_mant_signed = local_acc_sign ? -$signed({1'b0, acc_mant_aligned}) : $signed({1'b0, acc_mant_aligned});
        mul_mant_signed = block_sign_pre ? -$signed({1'b0, mul_mant_aligned}) : $signed({1'b0, mul_mant_aligned});
        sum_mant = acc_mant_signed + mul_mant_signed;

        if (sum_mant[fp_mant_bits+2]) begin
            acc_sign_pre = 1'b1;
            sum_mant_abs = (fp_mant_bits + 2)'(-sum_mant[fp_mant_bits+1:0]);
        end else begin
            acc_sign_pre = 1'b0;
            sum_mant_abs = sum_mant[fp_mant_bits+1:0];
        end

        add_lead_idx = -1;
        normalized_mant = '0;
        normalized_exp = '0;
        acc_exp_pre = '0;
        acc_mant_pre = '0;

        for (int i = fp_mant_bits + 1; i >= 0; i--) begin
            if ((add_lead_idx == -1) && sum_mant_abs[i]) begin
                add_lead_idx = i;
            end
        end

        if (add_lead_idx >= 0) begin
            if (add_lead_idx > fp_mant_bits) begin
                normalized_mant = {1'b0, sum_mant_abs} >> (add_lead_idx - fp_mant_bits);
            end else begin
                normalized_mant = (fp_mant_bits + 1)'(sum_mant_abs) << (fp_mant_bits - add_lead_idx);
            end

            if (add_lead_idx > fp_mant_bits) begin
                normalized_exp = (fp_exp_bits + 1)'(max_exp) + (add_lead_idx - fp_mant_bits);
            end else begin
                if (max_exp >= (fp_mant_bits - add_lead_idx)) begin
                    normalized_exp = (fp_exp_bits + 1)'(max_exp) - (fp_mant_bits - add_lead_idx);
                end else begin
                    normalized_exp = '0;
                end
            end

            if (normalized_exp[fp_exp_bits]) begin
                acc_exp_pre = {fp_exp_bits{1'b1}};
                acc_mant_pre = {fp_mant_bits{1'b1}};
            end else begin
                acc_exp_pre = normalized_exp[fp_exp_bits-1:0];
                acc_mant_pre = normalized_mant[fp_mant_bits-1:0];
            end
        end else begin
            acc_sign_pre = 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Systolic pass-through: activation flows right, weight flows down
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            amxfp4_a_scale_pos_out <= '0;
            amxfp4_a_scale_neg_out <= '0;
            amxfp4_a_elem_out      <= '0;
        end else begin
            amxfp4_a_scale_pos_out <= amxfp4_a_scale_pos_in;
            amxfp4_a_scale_neg_out <= amxfp4_a_scale_neg_in;
            amxfp4_a_elem_out      <= amxfp4_a_elem_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            amxfp4_b_scale_pos_out <= '0;
            amxfp4_b_scale_neg_out <= '0;
            amxfp4_b_elem_out      <= '0;
        end else begin
            amxfp4_b_scale_pos_out <= amxfp4_b_scale_pos_in;
            amxfp4_b_scale_neg_out <= amxfp4_b_scale_neg_in;
            amxfp4_b_elem_out      <= amxfp4_b_elem_in;
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
            local_acc_sign <= acc_sign_pre;
            local_acc_exp  <= acc_exp_pre;
            local_acc_mant <= acc_mant_pre;
        end
    end

endmodule
