`timescale 1ns/1ps
// Normalize four exp values per cycle using the shared eight-stage FP32
// multiplier. Each row supplies a common reciprocal; II=1 when unstalled.
module softmax_norm4(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,
    input wire [127:0] s_data,input wire [31:0] s_scale,
    input wire [3:0] s_keep,input wire [9:0] s_tag,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_data,output wire [9:0] m_tag
);
    wire [3:0] unused_keep;
    softmax_fp32_mul4 u_mul(
        .clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_scale(s_scale),
        .s_keep(s_keep),.s_tag(s_tag),.m_valid(m_valid),.m_ready(m_ready),
        .m_data(m_data),.m_keep(unused_keep),.m_tag(m_tag));
endmodule
