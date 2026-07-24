module MXFP4_PLUS_PE #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter bm_idx_bits = $clog2(group_size)
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [shared_exp_bits-1:0] a_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] a_elem_in
    , input logic [bm_idx_bits-1:0] a_bm_idx_in
    , input logic [shared_exp_bits-1:0] b_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] b_elem_in
    , input logic [bm_idx_bits-1:0] b_bm_idx_in

    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    , output logic [shared_exp_bits-1:0] a_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] a_elem_out
    , output logic [bm_idx_bits-1:0] a_bm_idx_out
    , output logic [shared_exp_bits-1:0] b_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] b_elem_out
    , output logic [bm_idx_bits-1:0] b_bm_idx_out

    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    // Derived constants (must match MXFP4_PLUS_MUL)
    localparam PRODUCT_WIDTH = 9;
    localparam FIXED_SCALE = 2;
    localparam COMBINE_FRAC = 6;
    localparam LOG2_GROUP_SIZE = $clog2(group_size);
    localparam DOT_OUT_BITS = PRODUCT_WIDTH + LOG2_GROUP_SIZE;
    localparam COMBINE_BITS = DOT_OUT_BITS + (COMBINE_FRAC - FIXED_SCALE) + 3;
    localparam COMBINE_EXP_BITS = shared_exp_bits + 1;

    // MUL output (combined fixed-point dot product + block exponent)
    logic [COMBINE_EXP_BITS-1:0] mul_out_exp;
    logic signed [COMBINE_BITS-1:0] mul_out_sum;

    // Stage 2 (normalize) combinational signals
    logic block_sign_pre;
    logic [fp_exp_bits-1:0] block_exp_pre;
    logic [fp_mant_bits-1:0] block_mant_pre;
    logic [COMBINE_BITS-2:0] comb_abs;
    int   comb_lead_idx;
    logic [fp_mant_bits:0] comb_mant_full;
    logic [fp_exp_bits:0] block_exp_ext;

    // Local FP32 accumulator
    logic local_acc_sign;
    logic [fp_exp_bits-1:0] local_acc_exp;
    logic [fp_mant_bits-1:0] local_acc_mant;

    // FP32 addition signals
    logic [fp_exp_bits-1:0] exp_diff;
    logic acc_exp_larger;
    logic [fp_exp_bits-1:0] max_exp;
    logic [fp_mant_bits:0] acc_mant_ext;
    logic [fp_mant_bits:0] blk_mant_ext;
    logic [fp_mant_bits:0] acc_mant_aligned;
    logic [fp_mant_bits:0] blk_mant_aligned;
    logic signed [fp_mant_bits+1:0] acc_mant_signed;
    logic signed [fp_mant_bits+1:0] blk_mant_signed;
    logic signed [fp_mant_bits+2:0] sum_mant;
    logic [fp_mant_bits+1:0] sum_mant_abs;
    logic acc_sign_pre;
    logic [fp_exp_bits-1:0] acc_exp_pre;
    logic [fp_mant_bits-1:0] acc_mant_pre;
    logic [fp_mant_bits:0] normalized_mant;
    logic [fp_exp_bits:0] normalized_exp;
    int add_lead_idx;

    // ----------------------------------------------------------------
    // Stage 1: MX+ multiply (regular tree + BM compute + combine, registered)
    // ----------------------------------------------------------------
    MXFP4_PLUS_MUL #(
        .group_size(group_size)
        , .shared_exp_bits(shared_exp_bits)
        , .elem_bits(elem_bits)
        , .bm_idx_bits(bm_idx_bits)
    ) u_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .a_exp(a_exp_in)
        , .a_elem(a_elem_in)
        , .a_bm_idx(a_bm_idx_in)
        , .b_exp(b_exp_in)
        , .b_elem(b_elem_in)
        , .b_bm_idx(b_bm_idx_in)
        , .out_exp(mul_out_exp)
        , .out_sum(mul_out_sum)
    );

    // ----------------------------------------------------------------
    // Stage 2: normalize the combined fixed-point value to the custom FP32.
    // COMBINE_FRAC is already folded into mul_out_exp, so no further subtraction.
    // ----------------------------------------------------------------
    always_comb begin
        block_sign_pre = mul_out_sum[COMBINE_BITS-1];
        comb_abs = block_sign_pre ? (COMBINE_BITS-1)'(-mul_out_sum) : (COMBINE_BITS-1)'(mul_out_sum);

        comb_lead_idx  = -1;
        comb_mant_full = '0;
        block_exp_ext  = '0;
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

            block_exp_ext = (fp_exp_bits + 1)'(mul_out_exp) + (fp_exp_bits + 1)'(comb_lead_idx);
            if (block_exp_ext[fp_exp_bits]) begin
                block_exp_pre  = {fp_exp_bits{1'b1}};
                block_mant_pre = {fp_mant_bits{1'b1}};
            end else begin
                block_exp_pre  = block_exp_ext[fp_exp_bits-1:0];
                block_mant_pre = comb_mant_full[fp_mant_bits-1:0];
            end
        end
    end

    // ----------------------------------------------------------------
    // Stage 2 (continued): FP32 accumulation, fused with the normalize above
    // (identical backend structure to MXFP4_PE) -- no register between
    // normalize and accumulate, so cell sizing matches the MXFP4 baseline.
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
        blk_mant_ext = ((block_exp_pre == '0) && (block_mant_pre == '0)) ? '0 : {1'b1, block_mant_pre};

        if (acc_exp_larger) begin
            acc_mant_aligned = acc_mant_ext;
            blk_mant_aligned = blk_mant_ext >> exp_diff;
        end else begin
            acc_mant_aligned = acc_mant_ext >> exp_diff;
            blk_mant_aligned = blk_mant_ext;
        end

        acc_mant_signed = local_acc_sign ? -$signed({1'b0, acc_mant_aligned}) : $signed({1'b0, acc_mant_aligned});
        blk_mant_signed = block_sign_pre ? -$signed({1'b0, blk_mant_aligned}) : $signed({1'b0, blk_mant_aligned});
        sum_mant = acc_mant_signed + blk_mant_signed;

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
            a_exp_out    <= '0;
            a_elem_out   <= '0;
            a_bm_idx_out <= '0;
        end else begin
            a_exp_out    <= a_exp_in;
            a_elem_out   <= a_elem_in;
            a_bm_idx_out <= a_bm_idx_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_exp_out    <= '0;
            b_elem_out   <= '0;
            b_bm_idx_out <= '0;
        end else begin
            b_exp_out    <= b_exp_in;
            b_elem_out   <= b_elem_in;
            b_bm_idx_out <= b_bm_idx_in;
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
