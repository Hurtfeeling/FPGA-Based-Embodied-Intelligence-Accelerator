`timescale 1ns/1ps
// One command per direction, two reserved AXI bursts, same-ID ordered responses.
// Only burst lengths are queued; the payload still streams without a deep FIFO.
// TX has no elastic buffer: every TX handshake is an actual AXI W handshake.
// GEMM uses RX_ALWAYS_READY with a dedicated SRAM write port, including
// background BT loading while the independent read port feeds computation.
// The downstream pack/register/SRAM path is faster than the 128-bit AXI input.
// Generic users keep RX_ALWAYS_READY=0 to retain valid/ready flow control.
// RX only exposes beats whose RID/RESP/RLAST checks pass. A failed burst is
// drained through its observed RLAST before the command reports an error.
module gemm_axi_dma128 #(
    parameter integer C_M_AXI_ID_WIDTH     = 1,
    parameter integer C_M_AXI_AWUSER_WIDTH = 0,
    parameter integer C_M_AXI_ARUSER_WIDTH = 0,
    parameter integer C_M_AXI_WUSER_WIDTH  = 0,
    parameter integer C_M_AXI_RUSER_WIDTH  = 0,
    parameter integer C_M_AXI_BUSER_WIDTH  = 0,
    parameter integer MAX_BURST = 256,
    parameter integer RX_ALWAYS_READY = 0,
    parameter integer STRIDED_WRITE = 0
)(
    input  wire clk,
    input  wire rst_n,
    input  wire cmd_valid,
    output wire cmd_ready,
    input  wire cmd_write,
    input  wire [31:0] cmd_addr,
    input  wire [31:0] cmd_bytes,
    input wire cmd_strided,
    input wire [31:0] cmd_stride,
    output reg  done,
    output reg  error,
    output wire busy,
    output wire [127:0] rx_data,
    output wire rx_valid,
    input  wire rx_ready,
    input  wire [127:0] tx_data,
    input  wire tx_valid,
    output wire tx_ready,
	output wire [C_M_AXI_ID_WIDTH-1 : 0]        m_axi_awid,
	output wire [31 : 0]      m_axi_awaddr,
	output wire [7 : 0]                         m_axi_awlen,
	output wire [2 : 0]                         m_axi_awsize,
	output wire [1 : 0]                         m_axi_awburst,
	output wire                                 m_axi_awlock,
	output wire [3 : 0]                         m_axi_awcache,
	output wire [2 : 0]                         m_axi_awprot,
	output wire [3 : 0]                         m_axi_awqos,
	output wire [(C_M_AXI_AWUSER_WIDTH > 0 ? C_M_AXI_AWUSER_WIDTH : 1)-1 : 0]    m_axi_awuser,
	output wire                                 m_axi_awvalid,
	input  wire                                 m_axi_awready,
	output wire [128-1 : 0]      m_axi_wdata,
	output wire [128/8-1 : 0]    m_axi_wstrb,
	output wire                                 m_axi_wlast,
	output wire [(C_M_AXI_WUSER_WIDTH > 0 ? C_M_AXI_WUSER_WIDTH : 1)-1 : 0]     m_axi_wuser,
	output wire                                 m_axi_wvalid,
	input  wire                                 m_axi_wready,
	input  wire [C_M_AXI_ID_WIDTH-1 : 0]        m_axi_bid,
	input  wire [1 : 0]                         m_axi_bresp,
	input  wire [(C_M_AXI_BUSER_WIDTH > 0 ? C_M_AXI_BUSER_WIDTH : 1)-1 : 0]     m_axi_buser,
	input  wire                                 m_axi_bvalid,
	output wire                                 m_axi_bready,
	output wire [C_M_AXI_ID_WIDTH-1 : 0]        m_axi_arid,
	output wire [31 : 0]      m_axi_araddr,
	output wire [7 : 0]                         m_axi_arlen,
	output wire [2 : 0]                         m_axi_arsize,
	output wire [1 : 0]                         m_axi_arburst,
	output wire                                 m_axi_arlock,
	output wire [3 : 0]                         m_axi_arcache,
	output wire [2 : 0]                         m_axi_arprot,
	output wire [3 : 0]                         m_axi_arqos,
	output wire [(C_M_AXI_ARUSER_WIDTH > 0 ? C_M_AXI_ARUSER_WIDTH : 1)-1 : 0]    m_axi_aruser,
	output wire                                 m_axi_arvalid,
	input  wire                                 m_axi_arready,
	input  wire [C_M_AXI_ID_WIDTH-1 : 0]        m_axi_rid,
	input  wire [128-1 : 0]      m_axi_rdata,
	input  wire [1 : 0]                         m_axi_rresp,
	input  wire                                 m_axi_rlast,
	input  wire [(C_M_AXI_RUSER_WIDTH > 0 ? C_M_AXI_RUSER_WIDTH : 1)-1 : 0]     m_axi_ruser,
	input  wire                                 m_axi_rvalid,
	output wire                                 m_axi_rready
);
    // Allocation reserves response capacity before asserting AR/AW. A stalled
    // address remains valid and stable even after an error in an older burst.
    reg active,direction_write,failed,strided;
    reg addr_valid;
    reg [31:0] next_address,address,stride_q,next_row_address;
    reg [27:0] remaining_beats;
    reg [8:0] address_beats,beat_index;
    reg [3:0] row_left;
    reg [8:0] lengths[0:1];
    reg tail,data_head;
    reg [1:0] reserved,data_count,b_owed;
    wire [8:0] page_beats=9'd256-{1'b0,next_address[11:4]};
    wire request_strided=STRIDED_WRITE&&cmd_write&&cmd_strided;
    wire [37:0] strided_end={6'd0,cmd_addr}+({6'd0,cmd_stride}<<5)-{6'd0,cmd_stride}+38'd128;
    wire [32:0] command_end={1'b0,cmd_addr}+{1'b0,cmd_bytes};
    wire invalid_command=(cmd_bytes==0)||(cmd_bytes[3:0]!=0)||(cmd_addr[3:0]!=0)||
        (request_strided?((cmd_bytes!=4096)||(cmd_stride<128)||(cmd_stride[3:0]!=0)||
                          (strided_end>38'h100000000)):(command_end>33'h100000000));
    wire ar_fire=m_axi_arvalid&&m_axi_arready;
    wire aw_fire=m_axi_awvalid&&m_axi_awready;
    wire r_fire=m_axi_rvalid&&m_axi_rready;
    wire w_fire=m_axi_wvalid&&m_axi_wready;
    wire b_fire=m_axi_bvalid&&m_axi_bready;
    wire expected_last=(beat_index==lengths[data_head]-1'b1);
    wire r_bad=(m_axi_rid!=0)||(m_axi_rresp!=0)||(m_axi_rlast!=expected_last);
    wire r_fault=r_fire&&r_bad;
    wire b_fault=b_fire&&((m_axi_bid!=0)||(m_axi_bresp!=0));
    wire retire=direction_write?b_fire:(r_fire&&m_axi_rlast);
    wire data_pop=direction_write?(w_fire&&expected_last):(r_fire&&m_axi_rlast);
    wire allocate=active&&!failed&&!r_fault&&!b_fault&&!addr_valid&&
                  (remaining_beats!=0)&&(reserved<2);
    // 总剩余量先饱和到 256 拍；burst 选择只需比较 9 位数，避免把
    // 28 位 remaining 比较串到页边界、行边界及完整地址加法上。
    localparam [8:0] BURST_LIMIT=MAX_BURST;
    wire [8:0] remaining_limit=(|remaining_beats[27:8])?9'd256:{1'b0,remaining_beats[7:0]};
    wire [8:0] span_beats=(remaining_limit<BURST_LIMIT)?remaining_limit:BURST_LIMIT;
    wire [8:0] boundary_beats=(strided&&{5'd0,row_left}<page_beats)?{5'd0,row_left}:page_beats;
    wire [8:0] next_beats=(span_beats<boundary_beats)?span_beats:boundary_beats;
    // 每行固定 8 拍。下一行首地址在当前行发送前已独立准备好，行末
    // 直接选择该寄存器，不再走 burst长度 -> +bytes +stride -128 的长链。
    wire row_finishes=strided&&({5'd0,row_left}<=BURST_LIMIT)&&({5'd0,row_left}<=page_beats);
    wire page_finishes=(next_beats==page_beats);
    wire [19:0] following_page=next_address[31:12]+20'd1;
    // 页内加法只有 8 位；页号 +1 与长度选择并行计算，最后做选择。
    // burst 不跨 4 KiB，只有正好用完本页时才改变高位。
    wire [31:0] following_address={page_finishes?following_page:next_address[31:12],
        (next_address[11:4]+next_beats[7:0]),4'd0};
    assign busy=active;
    assign cmd_ready=rst_n&&!active;
    assign m_axi_arid=0;
    assign m_axi_araddr=address;
    assign m_axi_arlen=address_beats[7:0]-8'd1;
    assign m_axi_arsize=4;
    assign m_axi_arburst=1;
    assign m_axi_arlock=0;
    assign m_axi_arcache=4'b0011;
    assign m_axi_arprot=0;
    assign m_axi_arqos=0;
    assign m_axi_aruser=0;
    assign m_axi_arvalid=rst_n&&active&&!direction_write&&addr_valid;
    assign rx_data=m_axi_rdata;
    assign rx_valid=rst_n&&active&&!direction_write&&(data_count!=0)&&m_axi_rvalid&&!failed&&!r_bad;
    assign m_axi_rready=rst_n&&active&&!direction_write&&(data_count!=0)&&
        (failed||RX_ALWAYS_READY||rx_ready||(m_axi_rvalid&&r_bad));
    assign m_axi_awid=0;
    assign m_axi_awaddr=address;
    assign m_axi_awlen=address_beats[7:0]-8'd1;
    assign m_axi_awsize=4;
    assign m_axi_awburst=1;
    assign m_axi_awlock=0;
    assign m_axi_awcache=4'b0011;
    assign m_axi_awprot=0;
    assign m_axi_awqos=0;
    assign m_axi_awuser=0;
    assign m_axi_awvalid=rst_n&&active&&direction_write&&addr_valid;
    assign m_axi_wdata=tx_data;
    assign m_axi_wstrb=16'hffff;
    assign m_axi_wlast=expected_last;
    assign m_axi_wuser=0;
    // AW and W are independent, including W before a stalled AW handshake.
    assign m_axi_wvalid=rst_n&&active&&direction_write&&(data_count!=0)&&tx_valid;
    assign tx_ready=rst_n&&active&&direction_write&&(data_count!=0)&&m_axi_wready;
    assign m_axi_bready=rst_n&&active&&direction_write&&(b_owed!=0);
    always @(posedge clk)begin
        if(!rst_n)begin
            active<=0;direction_write<=0;failed<=0;strided<=0;addr_valid<=0;
            next_address<=0;address<=0;stride_q<=0;next_row_address<=0;remaining_beats<=0;
            address_beats<=0;beat_index<=0;row_left<=0;
            tail<=0;data_head<=0;reserved<=0;data_count<=0;b_owed<=0;
            done<=0;error<=0;
        end else begin
            done<=0;error<=0;
            if(cmd_valid&&cmd_ready)begin
                if(invalid_command)begin done<=1;error<=1;end
                else begin
                    active<=1;direction_write<=cmd_write;failed<=0;strided<=request_strided;
                    next_address<=cmd_addr;remaining_beats<=cmd_bytes[31:4];
                    stride_q<=cmd_stride;next_row_address<=cmd_addr+cmd_stride;row_left<=8;addr_valid<=0;
                    tail<=0;data_head<=0;reserved<=0;data_count<=0;b_owed<=0;beat_index<=0;
                end
            end
            if(active)begin
                if(ar_fire||aw_fire)addr_valid<=0;
                if(r_fault||b_fault)failed<=1;
                if(allocate)begin
                    address<=next_address;address_beats<=next_beats;addr_valid<=1;
                    lengths[tail]<=next_beats;tail<=!tail;
                    remaining_beats<=remaining_beats-next_beats;
                    if(row_finishes)begin
                        next_address<=next_row_address;
                        next_row_address<=next_row_address+stride_q;row_left<=8;
                    end else begin
                        next_address<=following_address;
                        if(strided)row_left<=row_left-next_beats[3:0];
                    end
                end
                case({allocate,retire})
                    2'b10:reserved<=reserved+1'b1;
                    2'b01:reserved<=reserved-1'b1;
                    default:begin end
                endcase
                case({allocate,data_pop})
                    2'b10:data_count<=data_count+1'b1;
                    2'b01:data_count<=data_count-1'b1;
                    default:begin end
                endcase
                case({aw_fire,b_fire})
                    2'b10:b_owed<=b_owed+1'b1;
                    2'b01:b_owed<=b_owed-1'b1;
                    default:begin end
                endcase
                if(data_pop)begin data_head<=!data_head;beat_index<=0;end
                else if(r_fire||w_fire)beat_index<=beat_index+1'b1;
                // Error completion waits for every reserved address/data/
                // response, including a VALID address stalled when fault hit.
                if(reserved==0&&!addr_valid&&(remaining_beats==0||failed))begin
                    active<=0;done<=1;error<=failed;
                end
            end
        end
    end
    initial begin
        if(MAX_BURST<1||MAX_BURST>256||C_M_AXI_ID_WIDTH<1)
            $fatal(1,"invalid GEMM AXI configuration");
    end
`ifndef SYNTHESIS
    always @(posedge clk)if(rst_n)begin
        if(reserved>2||data_count>2||b_owed>2)$fatal(1,"AXI descriptor credit overflow");
        if(RX_ALWAYS_READY&&rx_valid&&!rx_ready)$fatal(1,"unthrottled SRAM RX contract violated");
    end
`endif
endmodule
