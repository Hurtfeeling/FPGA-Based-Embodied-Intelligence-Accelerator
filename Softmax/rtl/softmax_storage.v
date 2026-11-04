`timescale 1ns/1ps
// Small elastic queue. Payload is deliberately not reset.
module softmax_fifo #(parameter WIDTH=128, DEPTH=32, AW=$clog2(DEPTH))(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,input wire [WIDTH-1:0] s_data,
    output wire m_valid,input wire m_ready,output wire [WIDTH-1:0] m_data
);
    reg [WIDTH-1:0] mem[0:DEPTH-1];
    reg [AW-1:0] wp,rp; reg [AW:0] count;
    wire push=s_valid&&s_ready,pop=m_valid&&m_ready;
    assign s_ready=count<DEPTH;
    assign m_valid=count!=0;
    assign m_data=mem[rp];
    always @(posedge clk) begin
        if(!rst_n||clear)begin wp<=0;rp<=0;count<=0;end
        else begin
            if(push)begin mem[wp]<=s_data;wp<=wp+1'b1;end
            if(pop)rp<=rp+1'b1;
            case({push,pop})
                2'b10:count<=count+1'b1;
                2'b01:count<=count-1'b1;
                default:begin end
            endcase
        end
    end
endmodule

// 16 KiB row buffer: four 1024x32 simple-dual-port BRAM banks.
// Bank n holds FP32 lane n of each 128-bit AXI beat. All banks share
// addresses/enables, providing one 128-bit read and one write per cycle.
// Synchronous read latency is one cycle; ren=0 holds the previous output.
// Do not reset the memory or read payload: validity is tracked by the reader.
module softmax_row_bram(
    input wire clk,we,ren,
    input wire [9:0] waddr,raddr,
    input wire [127:0] wdata,output wire [127:0] rdata
);
    genvar bank;
    generate for(bank=0;bank<4;bank=bank+1) begin:g_bank
        (* ram_style="block" *) reg [31:0] mem[0:1023];
        reg [31:0] rdata_q;
        always @(posedge clk) begin
            if(we)mem[waddr]<=wdata[bank*32+:32];
        end
        always @(posedge clk) begin
            if(ren)rdata_q<=mem[raddr];
        end
        assign rdata[bank*32+:32]=rdata_q;
    end endgenerate
`ifndef SYNTHESIS
    always @(posedge clk) if(we&&ren&&waddr==raddr)
        $fatal(1,"Softmax same-address BRAM read/write collision");
`endif
endmodule

// One-entry elastic register used to isolate the BRAM reader queue mux from the
// arithmetic pipeline. It is a real timing cut, not a combinational wire;
// payload and tag remain stable while downstream applies backpressure.
module softmax_stream_pipe128(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,input wire [127:0] s_data,
    input wire [9:0] s_tag,
    output wire m_valid,input wire m_ready,output wire [127:0] m_data,
    output wire [9:0] m_tag
);
    reg valid_q; reg [127:0] data_q; reg [9:0] tag_q;
    wire advance=!valid_q||m_ready;
    assign s_ready=advance&&rst_n&&!clear;
    assign m_valid=valid_q&&rst_n&&!clear;
    assign m_data=data_q; assign m_tag=tag_q;
    always @(posedge clk) begin
        if(!rst_n||clear)begin valid_q<=0;end
        else if(advance)begin
            valid_q<=s_valid;
            if(s_valid)begin data_q<=s_data;tag_q<=s_tag;end
        end
    end
endmodule

// Credit-controlled reader for a one-cycle synchronous BRAM. safe_count only
// advances when a returned word is safely copied into the private queue.
// Consequently the BRAM location can be reused even if the consumer stalls.
module softmax_row_reader #(parameter DEPTH=4, AW=$clog2(DEPTH))(
    input wire clk,rst_n,clear,start,input wire [10:0] words,
    output wire ram_en,output wire [9:0] ram_addr,
    input wire [127:0] ram_data,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_data,output wire [9:0] m_tag,
    output reg [10:0] safe_count,output reg busy,output reg done
);
    reg [137:0] queue[0:DEPTH-1];
    reg [AW-1:0] wp,rp;reg [AW:0] count;
    reg [10:0] issue_count,words_q;
    reg pending;reg [9:0] pending_tag;
    wire pop=m_valid&&m_ready;
    // No speculative credit from m_ready: a stalled consumer is always safe.
    assign ram_en=busy&&(issue_count<words_q)&&(({1'b0,count}+pending)<DEPTH)&&!clear;
    assign ram_addr=issue_count[9:0];
    assign m_valid=(count!=0)&&!clear;
    assign {m_tag,m_data}=queue[rp];
    always @(posedge clk)begin
        if(!rst_n||clear)begin
            wp<=0;rp<=0;count<=0;issue_count<=0;words_q<=0;
            pending<=0;pending_tag<=0;safe_count<=0;busy<=0;done<=0;
        end else begin
            done<=0;
            pending<=ram_en;
            if(ram_en)begin pending_tag<=ram_addr;issue_count<=issue_count+1'b1;end
            if(pending)begin
                queue[wp]<={pending_tag,ram_data};wp<=wp+1'b1;
                safe_count<=safe_count+1'b1;
            end
            if(pop)begin
                rp<=rp+1'b1;
                if({1'b0,m_tag}+11'd1==words_q)begin busy<=0;done<=1;end
            end
            case({pending,pop})
                2'b10:count<=count+1'b1;
                2'b01:count<=count-1'b1;
                default:begin end
            endcase
            if(start)begin
                wp<=0;rp<=0;count<=0;issue_count<=0;words_q<=words;
                pending<=0;safe_count<=0;busy<=1;done<=0;
            end
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk)if(rst_n&&!clear)begin
        if(pending&&count==DEPTH&&!pop)$fatal(1,"Softmax BRAM return overflow");
        if(start&&busy)$fatal(1,"Softmax reader restarted while busy");
        if(start&&(words==0||words>1024))$fatal(1,"Invalid BRAM row length");
    end
`endif
endmodule
