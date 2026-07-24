module FP16_MUL #(
    parameter fp16_bits = 16
    , parameter fp16_exp_bits = 5
    , parameter fp16_mant_bits = 10
    , parameter full_mant_bits = fp16_mant_bits + 1   // 11 (implicit 1 + 10 explicit)
    , parameter product_bits = 2 * full_mant_bits      // 22
    , parameter exp_sum_bits = fp16_exp_bits + 1       // 6
) (
    input logic clk
    , input logic rst_n

    , input logic [fp16_bits-1:0] fp16_a
    , input logic [fp16_bits-1:0] fp16_b

    , output logic out_sign
    , output logic [exp_sum_bits-1:0] out_exp_sum
    , output logic [product_bits-1:0] out_mant_product
);

    // FP16 decode
    logic sign_a, sign_b;
    logic [fp16_exp_bits-1:0] exp_a, exp_b;
    logic [fp16_mant_bits-1:0] mant_a, mant_b;
    logic [full_mant_bits-1:0] full_mant_a, full_mant_b;

    // Pre-register
    logic out_sign_pre;
    logic [exp_sum_bits-1:0] exp_sum_pre;
    logic [product_bits-1:0] product_pre;

    always_comb begin
        // Extract FP16 fields: {sign(1), exp(5), mant(10)}
        sign_a = fp16_a[15];
        exp_a  = fp16_a[14:10];
        mant_a = fp16_a[9:0];

        sign_b = fp16_b[15];
        exp_b  = fp16_b[14:10];
        mant_b = fp16_b[9:0];

        // Implicit leading 1 for normal (exp!=0), 0 for zero/subnormal
        full_mant_a = {(exp_a != '0), mant_a};
        full_mant_b = {(exp_b != '0), mant_b};

        // Product sign
        out_sign_pre = sign_a ^ sign_b;

        // Exponent sum (raw, no bias subtraction — implicit bias = 50)
        exp_sum_pre = exp_sum_bits'(exp_a) + exp_sum_bits'(exp_b);

        // Unsigned mantissa product (11-bit x 11-bit = 22-bit)
        product_pre = full_mant_a * full_mant_b;
    end

    // Register outputs (Stage 1)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_sign         <= '0;
            out_exp_sum      <= '0;
            out_mant_product <= '0;
        end else begin
            out_sign         <= out_sign_pre;
            out_exp_sum      <= exp_sum_pre;
            out_mant_product <= product_pre;
        end
    end

endmodule
