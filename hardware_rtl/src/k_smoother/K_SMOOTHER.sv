// One per output channel. Static prefill-mean K-smoothing:
//   PREFILL: acc += k_in  (FP32) over all prefill tokens
//   DECODE : k_out = k_in - mu  (frozen mu)
// mu is computed by ONE FP32 multiplier shared across channels (in K_SMOOTH_ARRAY)
// and written back here via mu_wdata/mu_we, so this unit holds no multiplier.
// A single FP32_ADDSUB is time-shared between accumulate (+) and decode subtract (-).
module K_SMOOTHER #(
    parameter fp_exp_bits = 8
    , parameter fp_mant_bits = 23
    , parameter W = 1 + fp_exp_bits + fp_mant_bits   // 32
) (
    input  logic clk
    , input  logic rst_n

    , input  logic acc_clr          // clear the accumulator (start of prefill)
    , input  logic k_valid          // accumulate k_in this cycle (prefill)
    , input  logic decode           // 1 = decode phase (compute k_out = k_in - mu)

    , input  logic [W-1:0] k_in
    , input  logic [W-1:0] mu_wdata // mu from the shared multiplier
    , input  logic mu_we            // write mu_reg this cycle

    , output logic [W-1:0] acc_q    // accumulator value (consumed by the shared multiplier)
    , output logic [W-1:0] k_out
    , output logic [W-1:0] mu_out
);

    logic [W-1:0] acc;
    logic [W-1:0] mu_reg;
    logic [W-1:0] k_out_r;

    // Shared add/sub: (acc + k_in) while accumulating, (k_in - mu) while decoding
    logic [W-1:0] addsub_a, addsub_b, addsub_y;
    logic         addsub_sub;

    always_comb begin
        addsub_a   = decode ? k_in   : acc;
        addsub_b   = decode ? mu_reg : k_in;
        addsub_sub = decode;          // subtract in decode, add in prefill
    end

    FP32_ADDSUB #(.fp_exp_bits(fp_exp_bits), .fp_mant_bits(fp_mant_bits)) u_addsub (
        .a(addsub_a), .b(addsub_b), .sub(addsub_sub), .y(addsub_y)
    );

    // FP32 accumulator
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)              acc <= '0;
        else if (acc_clr)        acc <= '0;
        else if (k_valid && !decode) acc <= addsub_y;
    end

    // Frozen mean register (written by the shared multiplier sequencer)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        mu_reg <= '0;
        else if (acc_clr)  mu_reg <= '0;
        else if (mu_we)    mu_reg <= mu_wdata;
    end

    // Registered decode output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        k_out_r <= '0;
        else if (decode)   k_out_r <= addsub_y;
    end

    assign acc_q  = acc;
    assign k_out  = k_out_r;
    assign mu_out = mu_reg;

endmodule
