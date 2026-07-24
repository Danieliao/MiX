// Self-checking testbench for INT8_PE (uniform INT8, group_size = 32).
//
// The PE computes a 32-element signed dot product into an INT32 accumulator.
// Reference model = the same dot product computed in the testbench, so the
// comparison is EXACT (integer arithmetic, no tolerance).
//
// Pipeline: stage 1 = MUL + adder tree (registered), stage 2 = accumulate
// (registered) -> the block result is visible 2 clocks after reset release.
module int8_tb;
    localparam int group_size = 32;
    localparam int data_bits  = 8;
    localparam int acc_bits   = 32;
    localparam int NVEC       = 1000;

    logic clk = 0, rst_n, acc_shift = 0;
    logic signed [acc_bits-1:0] acc_in = 0;

    logic signed [group_size-1:0][data_bits-1:0] a_in, b_in;
    logic signed [group_size-1:0][data_bits-1:0] a_out, b_out;
    logic signed [acc_bits-1:0] acc_out;

    INT8_PE #(.group_size(group_size), .data_bits(data_bits), .acc_bits(acc_bits)) dut (
        .clk(clk), .rst_n(rst_n), .acc_shift(acc_shift),
        .a_in(a_in), .b_in(b_in), .acc_in(acc_in),
        .a_out(a_out), .b_out(b_out), .acc_out(acc_out)
    );

    always #5 clk = ~clk;

    // random signed 8-bit operand in [-127, 127]
    function automatic logic signed [data_bits-1:0] rand_op();
        return data_bits'(($urandom() % 255) - 127);
    endfunction

    int seed = 32'h8A8B8C8D;

    initial begin
        int fails;
        longint ref_dot, dut_dot;
        logic signed [data_bits-1:0] a [group_size];
        logic signed [data_bits-1:0] b [group_size];

        void'($urandom(seed));
        fails = 0;

        for (int n = 0; n < NVEC; n++) begin
            // ---- stimulus + software reference ----
            ref_dot = 0;
            for (int i = 0; i < group_size; i++) begin
                a[i] = rand_op();
                b[i] = rand_op();
                a_in[i] = a[i];
                b_in[i] = b[i];
                ref_dot += longint'(a[i]) * longint'(b[i]);
            end

            // ---- drive one block with a cleared accumulator ----
            rst_n = 0; @(posedge clk); #1;
            rst_n = 1; @(posedge clk); @(posedge clk); #1;
            dut_dot = longint'(signed'(acc_out));

            if (ref_dot !== dut_dot) begin
                fails++;
                if (fails <= 8)
                    $display("  FAIL[%0d] ref=%0d dut=%0d", n, ref_dot, dut_dot);
            end
        end

        if (fails == 0) $display("PASS: all %0d vectors EXACT match", NVEC);
        else            $display("RESULT: %0d / %0d FAILED", fails, NVEC);
        $finish;
    end
endmodule
