module MXFP4_PE #(
    parameter group_size = 32
    , parameter shared_exp_bits = 8
    , parameter elem_bits = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
) (
    input logic clk
    , input logic rst_n

    , input logic acc_shift

    , input logic [shared_exp_bits-1:0] mxfp4_a_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem_in
    , input logic [shared_exp_bits-1:0] mxfp4_b_exp_in
    , input logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem_in

    , input logic acc_sign_in
    , input logic [fp_exp_bits-1:0] acc_exp_in
    , input logic [fp_mant_bits-1:0] acc_mant_in

    , output logic [shared_exp_bits-1:0] mxfp4_a_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] mxfp4_a_elem_out
    , output logic [shared_exp_bits-1:0] mxfp4_b_exp_out
    , output logic [group_size-1:0][elem_bits-1:0] mxfp4_b_elem_out

    , output logic acc_sign_out
    , output logic [fp_exp_bits-1:0] acc_exp_out
    , output logic [fp_mant_bits-1:0] acc_mant_out
);

    // E2M1 derived constants (must match MXFP4_MUL)
    localparam PRODUCT_WIDTH = 9;
    localparam FIXED_SCALE = 2;
    localparam LOG2_GROUP_SIZE = $clog2(group_size);
    localparam DOT_OUT_BITS = PRODUCT_WIDTH + LOG2_GROUP_SIZE;
    localparam MUL_EXP_BITS = shared_exp_bits + 1;

    // MUL outputs
    logic [MUL_EXP_BITS-1:0] mul_out_exp;
    logic signed [DOT_OUT_BITS-1:0] mul_out_sum;

    // MUL-to-FP32 normalization signals
    logic mul_sign_pre;
    logic [fp_exp_bits-1:0] mul_exp_pre;
    logic [fp_mant_bits-1:0] mul_mant_pre;

    logic [DOT_OUT_BITS-1:0] mul_sum_abs;
    logic [fp_mant_bits:0] mul_mant_full;
    logic [fp_exp_bits:0] mul_exp_ext;
    int mul_lead_idx;

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
    // Stage 1: MXFP4 multiply (E2M1 decode + adder tree, registered)
    // ----------------------------------------------------------------
    MXFP4_MUL #(
        .group_size(group_size)
        , .shared_exp_bits(shared_exp_bits)
        , .elem_bits(elem_bits)
    ) u_mxfp4_mul (
        .clk(clk)
        , .rst_n(rst_n)
        , .mxfp4_a_exp(mxfp4_a_exp_in)
        , .mxfp4_a_elem(mxfp4_a_elem_in)
        , .mxfp4_b_exp(mxfp4_b_exp_in)
        , .mxfp4_b_elem(mxfp4_b_elem_in)
        , .out_exp(mul_out_exp)
        , .out_sum(mul_out_sum)
    );

    // ----------------------------------------------------------------
    // Stage 2: Normalize MUL result to FP32 (identical to MXINT4_PE)
    // ----------------------------------------------------------------
    always_comb begin
        mul_sign_pre = mul_out_sum[DOT_OUT_BITS-1];
        mul_sum_abs = mul_sign_pre ? DOT_OUT_BITS'(~mul_out_sum + 1'b1) : DOT_OUT_BITS'(mul_out_sum);
        mul_lead_idx = -1;
        mul_mant_full = '0;
        mul_exp_ext = '0;
        mul_exp_pre = '0;
        mul_mant_pre = '0;

        for (int i = DOT_OUT_BITS - 1; i >= 0; i--) begin
            if ((mul_lead_idx == -1) && mul_sum_abs[i]) begin
                mul_lead_idx = i;
            end
        end

        if (mul_lead_idx >= 0) begin
            if (mul_lead_idx > fp_mant_bits) begin
                mul_mant_full = {1'b0, mul_sum_abs} >> (mul_lead_idx - fp_mant_bits);
            end else begin
                mul_mant_full = (fp_mant_bits + 1)'(mul_sum_abs) << (fp_mant_bits - mul_lead_idx);
            end

            mul_exp_ext = (fp_exp_bits + 1)'(mul_out_exp) + (fp_exp_bits + 1)'(mul_lead_idx);
            if (mul_exp_ext[fp_exp_bits]) begin
                mul_exp_pre = {fp_exp_bits{1'b1}};
                mul_mant_pre = {fp_mant_bits{1'b1}};
            end else begin
                mul_exp_pre = mul_exp_ext[fp_exp_bits-1:0];
                mul_mant_pre = mul_mant_full[fp_mant_bits-1:0];
            end
        end
    end

    // ----------------------------------------------------------------
    // FP32 accumulation (identical to MXINT4_PE)
    // ----------------------------------------------------------------
    always_comb begin
        if (local_acc_exp >= mul_exp_pre) begin
            exp_diff = local_acc_exp - mul_exp_pre;
            acc_exp_larger = 1'b1;
            max_exp = local_acc_exp;
        end else begin
            exp_diff = mul_exp_pre - local_acc_exp;
            acc_exp_larger = 1'b0;
            max_exp = mul_exp_pre;
        end

        acc_mant_ext = ((local_acc_exp == '0) && (local_acc_mant == '0)) ? '0 : {1'b1, local_acc_mant};
        mul_mant_ext = ((mul_exp_pre == '0) && (mul_mant_pre == '0)) ? '0 : {1'b1, mul_mant_pre};

        if (acc_exp_larger) begin
            acc_mant_aligned = acc_mant_ext;
            mul_mant_aligned = mul_mant_ext >> exp_diff;
        end else begin
            acc_mant_aligned = acc_mant_ext >> exp_diff;
            mul_mant_aligned = mul_mant_ext;
        end

        acc_mant_signed = local_acc_sign ? -$signed({1'b0, acc_mant_aligned}) : $signed({1'b0, acc_mant_aligned});
        mul_mant_signed = mul_sign_pre ? -$signed({1'b0, mul_mant_aligned}) : $signed({1'b0, mul_mant_aligned});
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
