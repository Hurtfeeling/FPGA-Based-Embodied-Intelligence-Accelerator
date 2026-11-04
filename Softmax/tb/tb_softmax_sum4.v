`timescale 1ns/1ps
module tb_softmax_sum4;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, clear = 0;
    reg s_valid = 0, s_last = 0;
    reg [127:0] s_data = 0;
    wire s_ready;
    wire m_valid;
    reg m_ready = 0;
    wire [63:0] m_sum;
    softmax_sum4 dut(.clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_last(s_last),
        .m_valid(m_valid),.m_ready(m_ready),.m_sum(m_sum));

    integer tests = 0, cycle = 0, accepted = 0;
    integer first_cycle = 0, last_cycle = 0;
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (rst_n && !clear && s_valid && s_ready) begin
            if (accepted == 0) first_cycle = cycle;
            accepted = accepted + 1;
            last_cycle = cycle;
        end
    end

    function automatic [127:0] pack4(input [31:0] a, b, c, d);
        pack4 = {d,c,b,a};
    endfunction

    // Drive changes at falling edges, keeping input stable across its handshake.
    task automatic beat(input [127:0] data, input last);
        begin
            @(negedge clk);
            s_valid = 1;
            s_data = data;
            s_last = last;
            @(posedge clk);
            while (!s_ready) @(posedge clk);
        end
    endtask

    task automatic idle(input integer cycles);
        begin
            @(negedge clk);
            s_valid = 0;
            s_last = 0;
            repeat (cycles) @(posedge clk);
        end
    endtask

    task automatic expect_result(input [63:0] expected, input integer stall);
        integer waited, i;
        begin
            idle(0);
            waited = 0;
            while (!m_valid && waited < 20) begin
                @(negedge clk);
                waited = waited + 1;
            end
            if (!m_valid) $fatal(1, "sum output timed out");
            if (cycle-last_cycle != 6)
                $fatal(1, "last input to m_valid latency: expected 6 cycles, got %0d", cycle-last_cycle);
            if (m_sum !== expected)
                $fatal(1, "sum mismatch: expected %016h, got %016h", expected, m_sum);
            if (s_ready) $fatal(1, "input accepted a new row before result consumed");
            for (i = 0; i < stall; i = i + 1) begin
                @(negedge clk);
                // Try to present another row during output backpressure.
                s_valid = 1;
                s_last = 1;
                s_data = {4{32'h3f800000}};
                if (!m_valid || m_sum !== expected || s_ready)
                    $fatal(1, "output changed or input reopened while stalled");
            end
            s_valid = 0;
            s_last = 0;
            m_ready = 1;
            @(posedge clk);
            @(negedge clk);
            m_ready = 0;
            if (m_valid || !s_ready) $fatal(1, "result handshake did not reset row state");
            tests = tests + 1;
            accepted = 0;
        end
    endtask

    integer i, j;
    reg [31:0] code[0:3];
    reg [63:0] ref_q[0:3], expected;
    integer shift, mantissa;
    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;

        // FP32 exp=[1,.75,.5,.25] gives a mathematical beat sum of 2.5.
        for (i = 0; i < 256; i = i + 1)
            beat(pack4(32'h3f800000,32'h3f400000,32'h3f000000,32'h3e800000),i==255);
        if (last_cycle-first_cycle != 255)
            $fatal(1, "four-lane sum did not sustain one beat per cycle");
        expect_result(64'd640 << 48, 17);

        // Maximum row sum is exactly 4096, requiring bit 60 of UQ16.48.
        for (i = 0; i < 1024; i = i + 1)
            beat({4{32'h3f800000}},i==1023);
        expect_result(64'd4096 << 48, 0);

        // RNE half-way values: .5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4.
        beat(pack4(32'h27000000,32'h27c00000,32'h28200000,32'h28600000),1);
        expect_result(64'd8, 2);
        // The two FP32 neighbors around the half-unit boundary.
        beat(pack4(32'h26ffffff,32'h27000001,32'h27000000,32'h00000000),1);
        expect_result(64'd1, 0);

        // Very small normals, subnormals and signed zero all quantize to 0.
        beat(pack4(32'h00000001,32'h007fffff,32'h00800000,32'h80000000),1);
        expect_result(64'd0, 0);

        // A finite dyadic set generated from a simple integer definition:
        // e=(2^23+mantissa)*2^(-23-shift); q is exact when shift<=25.
        // This reference contains no FP32 exponent decoding or CSA logic.
        expected = 0;
        for (i = 0; i < 257; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                shift = 1 + ((i*7+j*3)%25);
                mantissa = (i*7919+j*104729) & 23'h7fffff;
                code[j] = ((127-shift) << 23) | mantissa;
                ref_q[j] = (64'd8388608 + mantissa) << (25-shift);
                expected = expected + ref_q[j];
            end
            beat(pack4(code[0],code[1],code[2],code[3]),i==256);
            if (i%11==0 && i!=256) idle(3);
        end
        expect_result(expected, 5);

        // Clear while data is in the forward pipeline, then begin a clean row.
        beat({4{32'h3f800000}},0);
        @(negedge clk); clear=1; s_valid=0;
        @(negedge clk); clear=0;
        repeat (5) @(negedge clk);
        if (m_valid) $fatal(1, "clear produced a stale result");
        accepted=0;
        beat(pack4(32'h3f800000,0,0,0),1);
        expect_result(64'd1 << 48, 1);

        // Clear a stalled completed result, then process another independent row.
        beat({4{32'h3f800000}},1);
        idle(0);
        wait(m_valid);
        @(negedge clk); clear=1;
        @(negedge clk); clear=0;
        #1;
        if (m_valid || !s_ready) $fatal(1, "clear did not discard completed row");
        accepted=0;
        beat(pack4(32'h3f000000,32'h3f000000,0,0),1);
        expect_result(64'd1 << 48, 0);

        $display("PASS tb_softmax_sum4: %0d cases, II=1, max row, RNE, bubbles, stall and clear", tests);
        $finish;
    end
    initial begin
        #200000;
        $fatal(1, "tb_softmax_sum4 watchdog");
    end
endmodule
