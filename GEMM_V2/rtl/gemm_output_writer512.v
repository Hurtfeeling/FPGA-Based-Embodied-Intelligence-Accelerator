`timescale 1ns/1ps
// One 32x32 INT32 microtile, one linear or 32-row strided DMA command.
// Start the DDR command once the first committed 512-bit word exists.
// No RAM FIFO: two prefetch registers plus one unpack register hold at most
// three words. Credits include every issued read through its last W handshake.
// Abort fills missing words with zeros only to finish an accepted command.
module gemm_output_writer512 #(parameter integer C_HALF_MAJOR=0)(
    input wire clk,rst_n,cancel_all,
    input wire ctx_valid,output wire ctx_ready,
    input wire [31:0] ctx_job,ctx_m0,ctx_n0,ctx_ybase,ctx_ystride,
    input wire [11:0] ctx_cbase,
    input wire [6:0] produced_words,
    output wire c_req_valid,output wire [11:0] c_req_addr,
    input wire c_rsp_valid,input wire [511:0] c_rsp_data,
    output wire wr_cmd_valid,input wire wr_cmd_ready,
    output wire [31:0] wr_cmd_addr,wr_cmd_bytes,wr_cmd_stride,output wire wr_cmd_strided,
    input wire wr_done,wr_error,
    output wire [127:0] tx_data,output wire tx_valid,input wire tx_ready,
    output reg done,output wire fault,output wire busy,
    output wire [31:0] active_job
);
    localparam [2:0] S_IDLE=0,S_INIT=1,S_WAIT_WORD=2,S_ROW_CMD=3,
        S_ROW_WAIT=4,S_FINISH=5,S_ABORT=6;
    reg [2:0] state;
    reg active,row_started,contiguous;
    reg [31:0] job_q,row_ddr_addr,stride_q,init_rows,init_stride;
    reg [11:0] c_base;
    reg [4:0] init_step;
    reg [6:0] issued_words;
    reg [2:0] read_valid_q,read_zero_q;
    (* dont_touch="yes" *) reg c_req_valid_q;
    reg [11:0] c_req_addr_q;
    reg [1:0] buffered_words;
    reg [1:0] tx_part_q;
    reg head_valid_q,tail_valid_q;
    reg [511:0] head_data_q,tail_data_q;
    wire row_fault=active&&row_started&&wr_done&&wr_error;
    assign fault=row_fault||(active&&state>S_ABORT);
    assign busy=active;
    assign active_job=job_q;
    assign ctx_ready=!active&&!cancel_all;
    wire [6:0] drain_words=7'd64;
    wire [5:0] physical_word=C_HALF_MAJOR?{issued_words[0],issued_words[5:1]}:issued_words[5:0];
    // Use registered credits only: WREADY does not feed the SRAM read enable.
    wire read_issue=active&&!row_fault&&(state!=S_ABORT)&&(buffered_words<3)&&
        (issued_words<(cancel_all?(row_started?drain_words:7'd0):7'd64))&&
        ((({1'b0,physical_word}<produced_words)&&(issued_words<64))||(cancel_all&&row_started));
    wire read_zero=cancel_all&&({1'b0,physical_word}>=produced_words);
    assign c_req_valid=c_req_valid_q;
    assign c_req_addr=c_req_addr_q;
    assign wr_cmd_valid=active&&(state==S_ROW_CMD)&&!cancel_all&&!fault;
    assign wr_cmd_addr=row_ddr_addr;
    assign wr_cmd_bytes=32'd4096;
    assign wr_cmd_strided=!contiguous;
    assign wr_cmd_stride=stride_q;
    wire unpack_ready,unpack_empty;
    wire stream_flush=!active||(state==S_ABORT)||row_fault;
    wire stream_empty=!head_valid_q&&!tail_valid_q&&unpack_empty&&
                      (read_valid_q==0)&&(buffered_words==0);
    wire response_push=read_valid_q[2]&&!stream_flush;
    wire head_pop=head_valid_q&&unpack_ready;
    wire word_retire=tx_valid&&tx_ready&&(tx_part_q==3);
    wire [511:0] response_data=read_zero_q[2]?512'd0:c_rsp_data;
    gemm_unpack512_128 u_unpack(
        .clk(clk),.rst_n(rst_n),.flush(stream_flush),
        .s_data(head_data_q),.s_valid(head_valid_q),.s_ready(unpack_ready),
        .m_data(tx_data),.m_valid(tx_valid),.m_ready(tx_ready),.empty(unpack_empty));

    // The SRAM sees a local registered request, not the global cancel/credit
    // comparison. An already-issued read after a fault is harmless payload.
    always @(posedge clk)begin
        if(read_issue&&!read_zero)c_req_addr_q<=c_base+{6'd0,physical_word};
        if(!rst_n||stream_flush)c_req_valid_q<=0;
        else c_req_valid_q<=read_issue&&!read_zero;
    end
    // Only valid bits/credits reset. A late synchronous response always has a
    // reserved register even if DDR stops immediately after the read request.
    always @(posedge clk)begin
        if(head_pop)begin
            if(tail_valid_q)begin
                head_data_q<=tail_data_q;
                if(response_push)tail_data_q<=response_data;
            end else if(response_push)head_data_q<=response_data;
        end else if(response_push)begin
            if(!head_valid_q)head_data_q<=response_data;
            else tail_data_q<=response_data;
        end
    end
    always @(posedge clk)begin
        if(!rst_n||stream_flush)begin
            head_valid_q<=0;tail_valid_q<=0;buffered_words<=0;tx_part_q<=0;
        end else begin
            case({response_push,head_pop})
                2'b10:if(!head_valid_q)head_valid_q<=1;else tail_valid_q<=1;
                2'b01:begin head_valid_q<=tail_valid_q;tail_valid_q<=0;end
                2'b11:head_valid_q<=1;
                default:begin end
            endcase
            case({read_issue,word_retire})
                2'b10:buffered_words<=buffered_words+1'b1;
                2'b01:buffered_words<=buffered_words-1'b1;
                default:begin end
            endcase
            if(tx_valid&&tx_ready)tx_part_q<=tx_part_q+1'b1;
        end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=S_IDLE;active<=0;row_started<=0;contiguous<=0;
            job_q<=0;row_ddr_addr<=0;stride_q<=0;init_rows<=0;init_stride<=0;
            c_base<=0;init_step<=0;issued_words<=0;
            read_valid_q<=0;read_zero_q<=0;done<=0;
        end else begin
            done<=0;
            if(stream_flush)begin read_valid_q<=0;read_zero_q<=0;end
            else begin read_valid_q<={read_valid_q[1:0],read_issue};read_zero_q<={read_zero_q[1:0],read_zero};end
            if(read_issue)issued_words<=issued_words+1'b1;
            if(ctx_valid&&ctx_ready)begin
                active<=1;row_started<=0;job_q<=ctx_job;c_base<=ctx_cbase;
                row_ddr_addr<=ctx_ybase+(ctx_n0<<2);stride_q<=ctx_ystride;
                init_rows<=ctx_m0;init_stride<=ctx_ystride;init_step<=0;
                issued_words<=0;contiguous<=ctx_ystride==128;
                state<=S_INIT;
            end
            case(state)
                S_IDLE:begin end
                S_INIT:if(cancel_all)state<=S_ABORT;else begin
                    if(init_rows[0])row_ddr_addr<=row_ddr_addr+init_stride;
                    init_rows<=init_rows>>1;init_stride<=init_stride<<1;
                    if(init_step==31)state<=S_WAIT_WORD;else init_step<=init_step+1'b1;
                end
                S_WAIT_WORD:if(cancel_all)state<=S_ABORT;
                    else if(produced_words>0)state<=S_ROW_CMD;
                S_ROW_CMD:if(cancel_all)state<=S_ABORT;
                    else if(wr_cmd_ready)begin row_started<=1;state<=S_ROW_WAIT;end
                S_ROW_WAIT:if(wr_done)begin
                    row_started<=0;
                    if(cancel_all||wr_error)state<=S_ABORT;
                    else state<=S_FINISH;
                end
                S_FINISH:if(cancel_all)state<=S_ABORT;
                    else if(stream_empty)begin active<=0;done<=1;state<=S_IDLE;end
                S_ABORT:begin active<=0;row_started<=0;state<=S_IDLE;end
                default:state<=S_ABORT;
            endcase
            if(row_fault)begin row_started<=0;state<=S_ABORT;end
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk)if(rst_n&&!stream_flush)begin
        if(response_push&&head_valid_q&&tail_valid_q&&!head_pop)
            $fatal(1,"C prefetch registers overflowed reserved read credits");
        if(buffered_words>3||(word_retire&&buffered_words==0))
            $fatal(1,"C read credit accounting failed");
        if(read_valid_q[2]&&!read_zero_q[2]&&!c_rsp_valid)$fatal(1,"C read response latency mismatch");
    end
`endif
endmodule
