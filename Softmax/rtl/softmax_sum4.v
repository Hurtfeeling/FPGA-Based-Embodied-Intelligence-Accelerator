`timescale 1ns/1ps
// Verilog-2005. Four nonnegative FP32 exp results per beat, each in [0,1].
// Sum uses an unsigned 64-bit integer with 48 fractional bits.  FP32
// conversion is RNE; subnormals and values strictly below 2^-49 round to 0.
// Each row contains at most 1024 beats (4096 elements).  Output is stable
// under backpressure. A result handshake starts a fresh row automatically.
module softmax_sum4 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         s_valid,
    output wire         s_ready,
    input  wire [127:0] s_data,
    input  wire         s_last,
    output reg          m_valid,
    input  wire         m_ready,
    output reg  [63:0]  m_sum
);
    // Conversion of a valid, nonnegative exp result. exp<=1 bounds the
    // left shift to 25; 64 bits also leave room for the entire row sum.
    function automatic [63:0] fp32_to_uq48(input [31:0] x);
        reg [7:0] exponent;
        reg [23:0] mantissa, low_mask;
        reg [63:0] truncated;
        reg guard_bit, sticky_bit;
        integer shift;
        begin
            exponent = x[30:23];
            mantissa = {1'b1, x[22:0]};
            low_mask = 0;
            truncated = 0;
            guard_bit = 0;
            sticky_bit = 0;
            shift = 0;
            fp32_to_uq48 = 0;
            if (!x[31] && exponent >= 8'd78 && exponent <= 8'd127) begin
                if (exponent >= 8'd102) begin
                    fp32_to_uq48 = {40'd0, mantissa} << (exponent - 8'd102);
                end else begin
                    shift = 102 - exponent;
                    truncated = {40'd0, mantissa} >> shift;
                    guard_bit = mantissa[shift-1];
                    low_mask = (24'd1 << (shift-1)) - 24'd1;
                    sticky_bit = |(mantissa & low_mask);
                    fp32_to_uq48 = truncated +
                        ((guard_bit && (sticky_bit || truncated[0])) ? 64'd1 : 64'd0);
                end
            end
        end
    endfunction

    function automatic [63:0] csa_sum(input [63:0] a, b, c);
        csa_sum = a ^ b ^ c;
    endfunction
    function automatic [63:0] csa_carry(input [63:0] a, b, c);
        csa_carry = ((a & b) | (a & c) | (b & c)) << 1;
    endfunction

    reg input_closed;
    reg [2:0] convert_valid,convert_last;
    reg [49:0] shift0[0:3],shift1[0:3],shift2[0:3];
    reg [5:0] amount0[0:3];
    reg [2:0] amount1[0:3];
    reg sticky1[0:3],sticky2[0:3];
    // Keep the guard bit separately from sticky through two short barrels.
    // A mantissa at exponent 127 contributes mantissa<<25 to UQ16.48.
    function automatic [50:0] shift_low(input [49:0] a,input [2:0] n);
        reg [49:0] x;reg lost;
        begin
            x=a;lost=0;
            if(n[0])begin lost=lost|x[0];x=x>>1;end
            if(n[1])begin lost=lost|(|x[1:0]);x=x>>2;end
            if(n[2])begin lost=lost|(|x[3:0]);x=x>>4;end
            shift_low={lost,x};
        end
    endfunction
    function automatic [50:0] shift_high(input [49:0] a,input [2:0] n,input sticky);
        reg [49:0] x;reg lost;
        begin
            x=a;lost=sticky;
            if(n[0])begin lost=lost|(|x[7:0]);x=x>>8;end
            if(n[1])begin lost=lost|(|x[15:0]);x=x>>16;end
            if(n[2])begin lost=lost|(|x[31:0]);x=x>>32;end
            shift_high={lost,x};
        end
    endfunction
    wire [63:0] converted[0:3];
    genvar cv;
    generate for(cv=0;cv<4;cv=cv+1)begin:g_convert
        assign converted[cv]={15'b0,shift2[cv][49:1]}+
            (shift2[cv][0]&&(sticky2[cv]||shift2[cv][1]));
    end endgenerate
    integer c;
    reg p0_valid, p0_last;
    reg [63:0] q0, q1, q2, q3;
    reg p1_valid, p1_last;
    reg [63:0] u_q, v_q;
    reg [63:0] acc_a, acc_b;
    reg finish_pending;

    wire [63:0] fwd_sum = csa_sum(q0, q1, q2);
    wire [63:0] fwd_carry = csa_carry(q0, q1, q2);
    wire [63:0] next_u = csa_sum(fwd_sum, fwd_carry, q3);
    wire [63:0] next_v = csa_carry(fwd_sum, fwd_carry, q3);

    // Two 3:2 compression levels in the feedback path, without a wide
    // carry-propagation chain. acc_a+acc_b denotes the running integer sum.
    wire [63:0] feedback_sum = csa_sum(acc_a, acc_b, u_q);
    wire [63:0] feedback_carry = csa_carry(acc_a, acc_b, u_q);
    wire [63:0] next_a = csa_sum(feedback_sum, feedback_carry, v_q);
    wire [63:0] next_b = csa_carry(feedback_sum, feedback_carry, v_q);

    assign s_ready = rst_n && !clear && !input_closed && !m_valid;
    wire accept = s_valid && s_ready;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            input_closed <= 0;
            convert_valid<=0;convert_last<=0;
            p0_valid <= 0;
            p0_last <= 0;
            p1_valid <= 0;
            p1_last <= 0;
            finish_pending <= 0;
            acc_a <= 0;
            acc_b <= 0;
            m_valid <= 0;
            m_sum <= 0;
        end else begin
            convert_valid<={convert_valid[1:0],accept};
            convert_last<={convert_last[1:0],s_last};
            if (accept) begin
                for(c=0;c<4;c=c+1)begin
                    shift0[c]<=(!s_data[c*32+31]&&s_data[c*32+23+:8]>=8'd78&&
                        s_data[c*32+23+:8]<=8'd127)?{1'b1,s_data[c*32+:23],26'b0}:50'b0;
                    amount0[c]<=8'd127-s_data[c*32+23+:8];
                end
                if (s_last) input_closed <= 1;
            end
            for(c=0;c<4;c=c+1)begin
                {sticky1[c],shift1[c]}<=shift_low(shift0[c],amount0[c][2:0]);
                amount1[c]<=amount0[c][5:3];
                {sticky2[c],shift2[c]}<=shift_high(shift1[c],amount1[c],sticky1[c]);
            end
            p0_valid<=convert_valid[2];
            if(convert_valid[2])begin
                q0<=converted[0];q1<=converted[1];q2<=converted[2];q3<=converted[3];
                p0_last<=convert_last[2];
            end

            p1_valid <= p0_valid;
            if (p0_valid) begin
                u_q <= next_u;
                v_q <= next_v;
                p1_last <= p0_last;
            end

            if (p1_valid) begin
                acc_a <= next_a;
                acc_b <= next_b;
                if (p1_last) finish_pending <= 1;
            end

            // The final beat has reached acc_a/acc_b on the previous edge.
            // Only this row-boundary operation propagates carry across 64 bits.
            if (finish_pending) begin
                m_sum <= acc_a + acc_b;
                m_valid <= 1;
                finish_pending <= 0;
            end

            if (m_valid && m_ready) begin
                m_valid <= 0;
                input_closed <= 0;
                acc_a <= 0;
                acc_b <= 0;
            end
        end
    end

`ifndef SYNTHESIS
    integer lane;
    reg [10:0] accepted_beats;
    always @(posedge clk) begin
        if (!rst_n || clear || (m_valid && m_ready)) accepted_beats <= 0;
        else if (accept) begin
            if (accepted_beats >= 1024)
                $fatal(1, "softmax_sum4 row exceeds 4096 FP32 values");
            accepted_beats <= accepted_beats + 1'b1;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if ((s_data[lane*32+31] && |s_data[lane*32+:31]) ||
                    (s_data[lane*32+:31] > 31'h3f800000))
                    $fatal(1, "softmax_sum4 expects finite exp in [0,1]");
            end
        end
    end
`endif
endmodule
