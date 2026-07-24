// Self-checking TB for K_SMOOTH_ARRAY. Per channel: stream N random FP32 keys
// (prefill), form mu = sum*(1/N), then check mu ≈ channel mean and decode keys
// k_out ≈ k_in - mu. FP32 here is IEEE-754 single, so we use shortreal for the
// reference (the HW truncates / flushes subnormals → small tolerance).
module k_smooth_tb;
    localparam CH = 4, W = 32;
    localparam int NMAX = 256;

    logic clk = 0, rst_n;
    logic acc_clr, k_valid, mu_load, decode;
    logic [CH-1:0][W-1:0] k_in;
    logic [W-1:0] inv_tc;
    logic [CH-1:0][W-1:0] k_out, mu_out;
    logic mu_valid;

    K_SMOOTH_ARRAY #(.channel_num(CH)) dut (
        .clk(clk), .rst_n(rst_n),
        .acc_clr(acc_clr), .k_valid(k_valid), .mu_load(mu_load), .decode(decode),
        .k_in(k_in), .inv_token_count(inv_tc),
        .k_out(k_out), .mu_out(mu_out), .mu_valid(mu_valid)
    );

    always #5 clk = ~clk;

    function automatic logic [31:0] f2b(input shortreal x); return $shortrealtobits(x); endfunction
    function automatic shortreal b2f(input logic [31:0] b); return $bitstoshortreal(b); endfunction

    shortreal keys [CH][NMAX];
    int seed = 32'hC0FFEE;
    int total_fail = 0;

    // run one (channel-biased) prefill of N tokens + mu check + a few decode checks
    task automatic run_test(input int N);
        real bias, noise, sum [CH], mean [CH], got, exp_v, rel, tol;
        shortreal dk [CH];
        int fails;
        fails = 0;

        // generate keys: each channel has a large per-channel bias + small noise (like K)
        for (int c = 0; c < CH; c++) begin
            sum[c] = 0.0;
            bias = real'((c - 1) * 200);                 // -200, 0, 200, 400
            for (int t = 0; t < N; t++) begin
                noise = real'((($urandom() % 4001) - 2000)) / 100.0;   // ±20
                keys[c][t] = shortreal'(bias + noise);
                sum[c] += real'(keys[c][t]);
            end
            mean[c] = sum[c] / real'(N);
        end

        // ---- prefill: clear, then stream N keys ----
        acc_clr = 1; k_valid = 0; mu_load = 0; decode = 0; inv_tc = f2b(1.0);
        @(posedge clk); #1; acc_clr = 0;
        for (int t = 0; t < N; t++) begin
            for (int c = 0; c < CH; c++) k_in[c] = f2b(keys[c][t]);
            k_valid = 1;
            @(posedge clk); #1;
        end
        k_valid = 0;

        // ---- form mu = acc * (1/N) via the shared sequential multiplier ----
        inv_tc = f2b(shortreal'(1.0 / real'(N)));
        mu_load = 1; @(posedge clk); #1; mu_load = 0;
        while (!mu_valid) begin @(posedge clk); #1; end

        // ---- check mu ≈ mean ----
        for (int c = 0; c < CH; c++) begin
            got = real'(b2f(mu_out[c]));
            exp_v = mean[c];
            rel = (exp_v != 0.0) ? ((got - exp_v) / exp_v) : (got - exp_v);
            if (rel < 0) rel = -rel;
            tol = 1e-3;
            if (rel > tol) begin
                fails++;
                if (total_fail + fails <= 12)
                    $display("  MU FAIL N=%0d ch=%0d: got=%0.6f mean=%0.6f rel=%0.2e", N, c, got, exp_v, rel);
            end
        end

        // ---- decode: a few keys, check k_out = k_in - mu ----
        decode = 1;
        for (int s = 0; s < 4; s++) begin
            for (int c = 0; c < CH; c++) begin
                dk[c] = shortreal'(real'((c-1)*200) + real'((($urandom() % 4001) - 2000)) / 100.0);
                k_in[c] = f2b(dk[c]);
            end
            @(posedge clk); #1;   // k_out registers k_in - mu
            for (int c = 0; c < CH; c++) begin
                got = real'(b2f(k_out[c]));
                exp_v = real'(dk[c]) - real'(b2f(mu_out[c]));
                rel = (exp_v != 0.0) ? ((got - exp_v) / exp_v) : (got - exp_v);
                if (rel < 0) rel = -rel;
                tol = 1e-3 + 1e-3 / (((exp_v < 0.0) ? -exp_v : exp_v) + 1e-6); // looser near zero (cancellation)
                if (rel > tol && ((got - exp_v > 1e-2) || (exp_v - got > 1e-2))) begin
                    fails++;
                    if (total_fail + fails <= 12)
                        $display("  KOUT FAIL N=%0d ch=%0d: got=%0.6f exp=%0.6f", N, c, got, exp_v);
                end
            end
        end
        decode = 0;

        $display("[N=%0d] %0s (%0d fails)", N, (fails == 0) ? "PASS" : "FAIL", fails);
        total_fail += fails;
    endtask

    initial begin
        void'($urandom(seed));
        rst_n = 0; acc_clr = 0; k_valid = 0; mu_load = 0; decode = 0;
        k_in = '0; inv_tc = f2b(1.0);
        @(posedge clk); #1; rst_n = 1;

        run_test(16);    // power of two
        run_test(64);    // power of two
        run_test(100);   // non power of two (1/100 not exact in fp32)
        run_test(250);   // larger, non power of two

        if (total_fail == 0) $display("PASS: all K-smooth tests within tolerance");
        else                 $display("RESULT: %0d total failures", total_fail);
        $finish;
    end
endmodule
