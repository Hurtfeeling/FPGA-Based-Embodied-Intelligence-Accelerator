`timescale 1ns/1ps
// 独立综合整个 A/BT/C 容量，不修改已有 Vivado 工程或 bitstream。
module gemm_v2_memories(
    input wire clk,
    input wire [2:0] we,ren,
    input wire [13:0] a_waddr,a_raddr,b_waddr,b_raddr,
    input wire [11:0] c_waddr,c_raddr,
    input wire [255:0] a_wdata,b_wdata,
    input wire [511:0] c_wdata,
    output wire [255:0] a_rdata,b_rdata,
    output wire [511:0] c_rdata
);
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1),.MIXED_RAM(1)) u_a(
        .clk(clk),.we(we[0]),.ren(ren[0]),.waddr(a_waddr),.raddr(a_raddr),.wdata(a_wdata),.rdata(a_rdata));
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1),.MIXED_RAM(1)) u_bt(
        .clk(clk),.we(we[1]),.ren(ren[1]),.waddr(b_waddr),.raddr(b_raddr),.wdata(b_wdata),.rdata(b_rdata));
    gemm_result_sram512 #(.RAM_STYLE("ultra")) u_c(
        .clk(clk),.we(we[2]),.ren(ren[2]),.waddr(c_waddr),.raddr(c_raddr),.wdata(c_wdata),.rdata(c_rdata));
endmodule
