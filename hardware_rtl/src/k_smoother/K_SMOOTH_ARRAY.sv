// channel_num independent K_SMOOTHER units (one per output channel), plus ONE
// shared FP32 multiplier that forms mu = acc * inv_token_count for each channel.
// Because mu is computed once per prefill (not throughput-critical), the multiplier
// is sequenced across the channels (one channel per cycle), so the array holds a
// single FP32_MUL instead of one per channel. channel_num = 4 matches the 4x4
// (block-32) arrays' column count.
module K_SMOOTH_ARRAY #(
    parameter channel_num = 4
    , parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter W = 1 + fp_exp_bits + fp_mant_bits   // 32
) (
    input  logic clk
    , input  logic rst_n

    , input  logic acc_clr
    , input  logic k_valid
    , input  logic mu_load      // pulse: start computing mu for all channels
    , input  logic decode

    , input  logic [channel_num-1:0][W-1:0] k_in
    , input  logic [W-1:0] inv_token_count

    , output logic [channel_num-1:0][W-1:0] k_out
    , output logic [channel_num-1:0][W-1:0] mu_out
    , output logic mu_valid     // all channels' mu computed and frozen
);

    localparam int CW = $clog2(channel_num + 1);

    logic [channel_num-1:0][W-1:0] acc_q;
    logic [channel_num-1:0] mu_we;
    logic [W-1:0] mul_a, mul_b, mul_y;

    // Sequencer: on mu_load, walk channels 0..channel_num-1, one per cycle,
    // writing each channel's mu_reg from the shared multiplier.
    logic running;
    logic [CW-1:0] mu_cnt;
    logic mu_valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0; mu_cnt <= '0; mu_valid_r <= 1'b0;
        end else if (acc_clr) begin
            running <= 1'b0; mu_cnt <= '0; mu_valid_r <= 1'b0;
        end else if (mu_load && !running && !mu_valid_r) begin
            running <= 1'b1; mu_cnt <= '0;
        end else if (running) begin
            if (mu_cnt == CW'(channel_num - 1)) begin
                running    <= 1'b0;
                mu_valid_r <= 1'b1;
            end
            mu_cnt <= mu_cnt + 1'b1;
        end
    end

    // Shared FP32 multiplier, time-multiplexed across channels
    always_comb begin
        mul_a = (mu_cnt < CW'(channel_num)) ? acc_q[mu_cnt] : acc_q[0];
        mul_b = inv_token_count;
        for (int c = 0; c < channel_num; c++)
            mu_we[c] = running && (mu_cnt == CW'(c));
    end

    FP32_MUL #(.fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)) u_mul (
        .a(mul_a), .b(mul_b), .y(mul_y)
    );

    assign mu_valid = mu_valid_r;

    genvar c;
    generate
        for (c = 0; c < channel_num; c++) begin : ch
            K_SMOOTHER #(
                .fp_exp_bits(fp_exp_bits)
                , .fp_mant_bits(fp_mant_bits)
            ) u_smoother (
                .clk(clk)
                , .rst_n(rst_n)
                , .acc_clr(acc_clr)
                , .k_valid(k_valid)
                , .decode(decode)
                , .k_in(k_in[c])
                , .mu_wdata(mul_y)
                , .mu_we(mu_we[c])
                , .acc_q(acc_q[c])
                , .k_out(k_out[c])
                , .mu_out(mu_out[c])
            );
        end
    endgenerate

endmodule
