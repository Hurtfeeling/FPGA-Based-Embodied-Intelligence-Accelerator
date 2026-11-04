`timescale 1ns/1ps
// Four-stage, four-lane FP32 Softmax. Input/output BRAM roles are fixed.
// Reader queues protect returned data before a later row may overwrite BRAM.
module softmax_core #(
    parameter integer OVERLAP_STORE_EXP=1,
    parameter EXP_LUT_FILE="exp_lut.mem"
)(
    input wire clk,rst_n,start_pulse,clear_status_pulse,cfg_enable,
    input wire [31:0] cfg_src,cfg_dst,cfg_dim,
    output reg busy,done,error,output reg [3:0] error_code,
    output wire rd_cmd_valid,input wire rd_cmd_ready,
    output reg [31:0] rd_cmd_addr,output wire [31:0] rd_cmd_bytes,
    input wire rd_done,rd_error,rd_busy,
    output wire wr_cmd_valid,input wire wr_cmd_ready,
    output reg [31:0] wr_cmd_addr,output wire [31:0] wr_cmd_bytes,
    input wire wr_done,wr_error,wr_busy,
    input wire [127:0] rx_data,input wire rx_valid,output wire rx_ready,
    output wire [127:0] tx_data,output wire tx_valid,input wire tx_ready
);
    `include "softmax_fp32_funcs.vh"
    localparam L_IDLE=0,L_CMD=1,L_DATA=2;
    localparam C_IDLE=0,C_WAIT=1,C_EXP=2,C_SUM=3,C_RECIP_REQ=4,
               C_RECIP_WAIT=5,C_NORM_WAIT=6,C_NORM=7,C_FINISH=8,C_CONVERT=9;
    localparam W_IDLE=0,W_CMD=1,W_DATA=2;
    reg [1:0] load_state,store_state;
    reg [3:0] calc_state;
    reg failed;
    reg [31:0] src_base,dst_base,row_bytes;
    reg [31:0] next_rd_addr,next_wr_addr;
    reg [2:0] cfg_phase;
    reg cfg_basic_bad;
    (* use_dsp="yes" *) reg [27:0] cfg_word_count;
    reg [31:0] cfg_byte_count;
    reg [32:0] cfg_src_end_q,cfg_dst_end_q;
    reg [16:0] rows,cols,calc_row,next_load_row,load_row,store_row,stored_rows;
    reg [10:0] words,load_count,exp_written,norm_written;
    reg input_ready,load_guard,load_dma_done,load_any,input_reclaim_started;
    reg [31:0] load_max,input_max,calc_max,row_sum,reciprocal;
    // Four independent lane maxima, followed by two registered merge levels.
    // Sortable IEEE keys allow an unsigned comparator per feedback path.
    // Drain all four stages before publishing input_max to the exp engine.
    reg [3:0] max_pipe_valid;
    reg [31:0] max_key[0:3],lane_max[0:3],max_pair0,max_pair1;
    reg output_pending;reg [16:0] pending_row;
    wire [16:0] cfg_rows={1'b0,cfg_dim[15:0]}+17'd1;
    wire [16:0] cfg_cols={1'b0,cfg_dim[31:16]}+17'd1;
    wire [63:0] cfg_stride=((64'd0+cfg_cols+3)>>2)<<4;
    wire bad_cfg=cfg_basic_bad||cfg_src_end_q>33'h100000000||cfg_dst_end_q>33'h100000000||
        (({1'b0,src_base}<cfg_dst_end_q)&&({1'b0,dst_base}<cfg_src_end_q));
    wire new_job=start_pulse&&!busy;
    wire math_clear=new_job||failed||!busy;

    wire rxq_valid,rxq_ready;wire [127:0] rxq_data;
    softmax_fifo u_rx_fifo(.clk(clk),.rst_n(rst_n),.clear(new_job),
        .s_valid(rx_valid),.s_ready(rx_ready),.s_data(rx_data),
        .m_valid(rxq_valid),.m_ready(rxq_ready),.m_data(rxq_data));
    wire [10:0] input_safe,store_safe;
    assign rxq_ready=failed||((load_state==L_DATA)&&
        (load_count<words)&&(!load_guard||load_count<input_safe));
    wire load_fire=rxq_valid&&rxq_ready&&(load_state==L_DATA);

    function automatic [3:0] keep_for(input [9:0] word_index,input [16:0] k);
        integer l;
        begin for(l=0;l<4;l=l+1)keep_for[l]=({5'd0,word_index,2'b00}+l)<k;end
    endfunction
    wire [3:0] load_keep=keep_for(load_count[9:0],cols);
    reg load_bad,load_any_next;
    integer lane;
    reg [31:0] value;
    function automatic [31:0] fp32_order_key(input [31:0] a);
        reg [31:0] aa;
        begin
            aa=fp32_flush(a);
            fp32_order_key=aa[31]?~aa:{1'b1,aa[30:0]};
        end
    endfunction
    wire [31:0] merged_max_key=(max_pair0>=max_pair1)?max_pair0:max_pair1;
    always @*begin
        load_bad=0;load_any_next=load_any;value=0;
        for(lane=0;lane<4;lane=lane+1)begin
            value=fp32_flush(rxq_data[lane*32+:32]);
            if(load_fire&&load_keep[lane])begin
                if(fp32_is_nan(value)||value==32'h7f800000)load_bad=1;
                else if(value!=32'hff800000)begin
                    load_any_next=1;
                end
            end
        end
    end
    // DMA completion may coincide with the last beat entering RX FIFO.  Do
    // not compare the BRAM write count until the FIFO has drained.
    wire load_complete=(load_state==L_DATA)&&load_dma_done&&!rxq_valid&&
        !(|max_pipe_valid)&&(load_count==words);
    wire fault_read=rd_error;
    wire fault_short=load_dma_done&&!rxq_valid&&(load_count!=words);
    wire fault_overrun=load_dma_done&&(load_count>=words)&&rxq_valid;
    wire fault_write=wr_error;
    wire fault_numeric=load_fire&&load_bad;
    wire fault_mask=load_complete&&!load_any_next;
    wire fault=fault_read||fault_short||fault_overrun||fault_write||fault_numeric||fault_mask;

    assign rd_cmd_valid=load_state==L_CMD;
    assign rd_cmd_bytes=row_bytes;
    assign wr_cmd_valid=store_state==W_CMD;
    assign wr_cmd_bytes=row_bytes;
    wire start_store=wr_cmd_valid&&wr_cmd_ready;
    wire previous_output_available=(calc_row==0)||(stored_rows>=calc_row)||
        (OVERLAP_STORE_EXP&&(store_state==W_DATA)&&store_row+17'd1==calc_row);
    wire begin_exp=busy&&!failed&&!fault&&(calc_state==C_WAIT)&&input_ready&&
        previous_output_available&&!output_pending;
    wire norm_reader_busy,store_reader_busy;
    wire begin_norm=busy&&!failed&&!fault&&(calc_state==C_NORM_WAIT)&&!store_reader_busy;

    wire in_ren;wire [9:0] in_raddr;wire [127:0] in_rdata;
    wire in_stream_valid,in_stream_ready;wire [127:0] in_stream_data;
    wire [9:0] in_stream_tag;wire in_reader_busy,in_reader_done;
    softmax_row_bram u_input(.clk(clk),.we(load_fire&&!failed&&!load_bad),
        .waddr(load_count[9:0]),.wdata(rxq_data),.ren(in_ren),.raddr(in_raddr),.rdata(in_rdata));
    softmax_row_reader u_input_reader(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .start(begin_exp),.words(words),.ram_en(in_ren),.ram_addr(in_raddr),.ram_data(in_rdata),
        .m_valid(in_stream_valid),.m_ready(in_stream_ready),.m_data(in_stream_data),
        .m_tag(in_stream_tag),.safe_count(input_safe),.busy(in_reader_busy),.done(in_reader_done));

    // Register the BRAM-reader return before the FP32 subtraction.  The
    // reader's queue address mux is otherwise directly on the exp critical
    // path and has very high fanout after BRAM inference.
    wire input_pipe_valid;
    wire [127:0] input_pipe_data; wire [9:0] input_pipe_tag;
    wire input_pipe_m_ready;
    wire exp_s_ready,exp_valid,exp_ready;wire [127:0] exp_data;wire [9:0] exp_tag;
    assign input_pipe_m_ready=exp_s_ready&&(calc_state==C_EXP)&&!failed;
    softmax_stream_pipe128 u_input_pipe(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(in_stream_valid),.s_ready(in_stream_ready),.s_data(in_stream_data),.s_tag(in_stream_tag),
        .m_valid(input_pipe_valid),.m_ready(input_pipe_m_ready),.m_data(input_pipe_data),.m_tag(input_pipe_tag));

    softmax_exp4 #(.LUT_FILE(EXP_LUT_FILE)) u_exp(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(input_pipe_valid&&(calc_state==C_EXP)&&!failed),.s_ready(exp_s_ready),
        .s_data(input_pipe_data),.s_max(calc_max),.s_keep(keep_for(input_pipe_tag,cols)),.s_tag(input_pipe_tag),
        .m_valid(exp_valid),.m_ready(exp_ready),.m_data(exp_data),.m_tag(exp_tag));

    reg exp_output_guard;
    wire output_space=!exp_output_guard||({1'b0,exp_tag}<store_safe);
    wire sum_s_ready,sum_valid,sum_ready;wire [63:0] sum_data;
    assign exp_ready=sum_s_ready&&output_space&&(calc_state==C_EXP)&&!failed;
    wire exp_fire=exp_valid&&exp_ready;
    softmax_sum4 u_sum(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(exp_valid&&output_space&&(calc_state==C_EXP)&&!failed),.s_ready(sum_s_ready),
        .s_data(exp_data),.s_last({1'b0,exp_tag}+11'd1==words),
        .m_valid(sum_valid),.m_ready(sum_ready),.m_sum(sum_data));
    wire sum_convert_ready,sum_float_valid;wire [31:0] sum_float_data;
    assign sum_ready=(calc_state==C_SUM)&&!failed&&sum_convert_ready;
    softmax_sum_to_fp32 u_sum_float(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(sum_valid&&(calc_state==C_SUM)&&!failed),.s_ready(sum_convert_ready),.s_sum(sum_data),
        .m_valid(sum_float_valid),.m_ready(calc_state==C_CONVERT&&!failed),.m_data(sum_float_data));
    wire reciprocal_s_ready,reciprocal_valid;
    wire [31:0] reciprocal_data;
    softmax_recip u_recip(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(calc_state==C_RECIP_REQ&&!failed),.s_ready(reciprocal_s_ready),.s_data(row_sum),
        .m_valid(reciprocal_valid),.m_ready(calc_state==C_RECIP_WAIT&&!failed),.m_data(reciprocal_data));

    wire norm_ren,store_ren;wire [9:0] norm_raddr,store_raddr;
    wire [127:0] out_rdata;
    wire norm_in_valid,norm_in_ready;wire [127:0] norm_in_data;wire [9:0] norm_in_tag;
    wire [10:0] norm_safe_unused;wire norm_reader_done,store_reader_done;
    softmax_row_reader u_norm_reader(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .start(begin_norm),.words(words),.ram_en(norm_ren),.ram_addr(norm_raddr),.ram_data(out_rdata),
        .m_valid(norm_in_valid),.m_ready(norm_in_ready),.m_data(norm_in_data),.m_tag(norm_in_tag),
        .safe_count(norm_safe_unused),.busy(norm_reader_busy),.done(norm_reader_done));
    wire norm_valid,norm_ready,norm_s_ready;wire [127:0] norm_data;wire [9:0] norm_tag;
    assign norm_in_ready=norm_s_ready&&(calc_state==C_NORM)&&!failed;
    assign norm_ready=(calc_state==C_NORM)&&!failed;
    softmax_norm4 u_norm(.clk(clk),.rst_n(rst_n),.clear(math_clear),
        .s_valid(norm_in_valid&&(calc_state==C_NORM)&&!failed),.s_ready(norm_s_ready),
        .s_data(norm_in_data),.s_scale(reciprocal),.s_keep(keep_for(norm_in_tag,cols)),.s_tag(norm_in_tag),
        .m_valid(norm_valid),.m_ready(norm_ready),.m_data(norm_data),.m_tag(norm_tag));
    wire norm_fire=norm_valid&&norm_ready;
    wire [9:0] store_tag_unused;
    softmax_row_reader #(.DEPTH(32)) u_store_reader(.clk(clk),.rst_n(rst_n),
        .clear(new_job||(failed&&store_state==W_IDLE&&!wr_busy)),
        .start(start_store),.words(words),.ram_en(store_ren),.ram_addr(store_raddr),.ram_data(out_rdata),
        .m_valid(tx_valid),.m_ready(tx_ready),.m_data(tx_data),.m_tag(store_tag_unused),
        .safe_count(store_safe),.busy(store_reader_busy),.done(store_reader_done));
    softmax_row_bram u_output(.clk(clk),.we(exp_fire||norm_fire),
        .waddr(norm_fire?norm_tag:exp_tag),.wdata(norm_fire?norm_data:exp_data),
        .ren(store_ren||norm_ren),.raddr(store_ren?store_raddr:norm_raddr),.rdata(out_rdata));

    integer max_lane;
    always @(posedge clk)begin
        if(!rst_n)begin
            busy<=0;done<=0;error<=0;error_code<=0;failed<=0;
            src_base<=0;dst_base<=0;row_bytes<=0;rows<=0;cols<=0;words<=0;
            cfg_phase<=0;cfg_basic_bad<=0;next_rd_addr<=0;next_wr_addr<=0;
            load_state<=L_IDLE;store_state<=W_IDLE;calc_state<=C_IDLE;
            calc_row<=0;next_load_row<=0;load_row<=0;store_row<=0;stored_rows<=0;
            load_count<=0;exp_written<=0;norm_written<=0;
            input_ready<=0;load_guard<=0;load_dma_done<=0;load_any<=0;input_reclaim_started<=0;
            load_max<=32'hff800000;input_max<=0;calc_max<=0;row_sum<=0;reciprocal<=0;
            max_pipe_valid<=0;max_pair0<=32'h007fffff;max_pair1<=32'h007fffff;
            for(max_lane=0;max_lane<4;max_lane=max_lane+1) lane_max[max_lane]<=32'h007fffff;
            output_pending<=0;pending_row<=0;exp_output_guard<=0;rd_cmd_addr<=0;wr_cmd_addr<=0;
        end else begin
            if(clear_status_pulse&&!busy)begin done<=0;error<=0;error_code<=0;failed<=0;end
            if(new_job)begin
                done<=0;error<=0;error_code<=0;failed<=0;busy<=1;
                cfg_phase<=1;
                cfg_basic_bad<=!cfg_enable||(cfg_cols>4096)||(|cfg_src[3:0])||(|cfg_dst[3:0]);
                src_base<=cfg_src;dst_base<=cfg_dst;row_bytes<=cfg_stride[31:0];
                next_rd_addr<=cfg_src;next_wr_addr<=cfg_dst;
                rows<=cfg_rows;cols<=cfg_cols;words<=cfg_stride[14:4];
                calc_row<=0;next_load_row<=0;stored_rows<=0;load_row<=0;store_row<=0;
                load_state<=L_IDLE;store_state<=W_IDLE;calc_state<=C_IDLE;
                input_ready<=0;input_reclaim_started<=0;load_dma_done<=0;load_any<=0;
                load_guard<=0;load_count<=0;output_pending<=0;exp_output_guard<=0;
                exp_written<=0;norm_written<=0;load_max<=32'hff800000;
                max_pipe_valid<=0;
                for(max_lane=0;max_lane<4;max_lane=max_lane+1) lane_max[max_lane]<=32'h007fffff;
            end else if(busy&&cfg_phase!=0)begin
                // One pipelined matrix-span check per START, off the normal
                // scheduler path. Keep byte carries through bit 32 for wrap.
                case(cfg_phase)
                    1:begin cfg_word_count<=words*rows;cfg_phase<=2;end
                    2:begin cfg_byte_count<={cfg_word_count,4'b0};cfg_phase<=3;end
                    3:begin
                        cfg_src_end_q<={1'b0,src_base}+{1'b0,cfg_byte_count};
                        cfg_dst_end_q<={1'b0,dst_base}+{1'b0,cfg_byte_count};cfg_phase<=4;
                    end
                    4:begin
                        cfg_phase<=0;
                        if(bad_cfg)begin busy<=0;error<=1;error_code<=1;end
                        else calc_state<=C_WAIT;
                    end
                    default:cfg_phase<=0;
                endcase
            end else if(busy)begin
                max_pipe_valid<={max_pipe_valid[2:0],load_fire&&!failed&&!load_bad};
                if(failed)max_pipe_valid<=0;
                for(max_lane=0;max_lane<4;max_lane=max_lane+1)begin
                    if(load_fire)max_key[max_lane]<=load_keep[max_lane]?
                        fp32_order_key(rxq_data[max_lane*32+:32]):32'h007fffff;
                    if(max_pipe_valid[0]&&max_key[max_lane]>lane_max[max_lane])
                        lane_max[max_lane]<=max_key[max_lane];
                end
                max_pair0<=(lane_max[0]>=lane_max[1])?lane_max[0]:lane_max[1];
                max_pair1<=(lane_max[2]>=lane_max[3])?lane_max[2]:lane_max[3];
                load_max<=merged_max_key[31]?{1'b0,merged_max_key[30:0]}:~merged_max_key;
                // Loader owns RX until its command and queued payload are drained.
                if(load_state==L_IDLE&&!failed&&!fault&&!input_ready&&
                    next_load_row<rows&&(next_load_row==0||input_reclaim_started))begin
                    load_state<=L_CMD;load_row<=next_load_row;
                    rd_cmd_addr<=next_rd_addr;
                    load_count<=0;load_dma_done<=0;load_guard<=next_load_row!=0;
                    load_any<=0;load_max<=32'hff800000;
                    max_pipe_valid<=0;
                    for(max_lane=0;max_lane<4;max_lane=max_lane+1) lane_max[max_lane]<=32'h007fffff;
                end
                if(rd_cmd_valid&&rd_cmd_ready)begin
                    load_state<=L_DATA;next_load_row<=next_load_row+1'b1;
                    next_rd_addr<=next_rd_addr+row_bytes;
                end
                if(rd_done)load_dma_done<=1;
                if(load_fire&&!failed)begin
                    load_count<=load_count+1'b1;load_any<=load_any_next;
                end
                if(load_complete&&!failed&&!fault)begin
                    input_ready<=1;input_max<=load_max;load_state<=L_IDLE;
                end
                if((failed||fault)&&(load_state==L_DATA)&&(load_dma_done||rd_done)&&!rxq_valid)
                    load_state<=L_IDLE;

                // The old output descriptor stays intact until its B responses.
                if(store_state==W_IDLE&&output_pending&&!failed&&!fault)begin
                    store_state<=W_CMD;store_row<=pending_row;
                    wr_cmd_addr<=next_wr_addr;output_pending<=0;
                end
                if(start_store)begin store_state<=W_DATA;next_wr_addr<=next_wr_addr+row_bytes;end
                if(wr_done)begin
                    store_state<=W_IDLE;
                    if(!wr_error)stored_rows<=stored_rows+1'b1;
                end

                if(begin_exp)begin
                    calc_state<=C_EXP;calc_max<=input_max;input_ready<=0;
                    input_reclaim_started<=1;exp_written<=0;
                    exp_output_guard<=OVERLAP_STORE_EXP&&(store_state==W_DATA)&&stored_rows<calc_row;
                end
                if(exp_fire)begin
                    exp_written<=exp_written+1'b1;
                    if({1'b0,exp_tag}+11'd1==words)calc_state<=C_SUM;
                end
                if(sum_valid&&sum_ready)calc_state<=C_CONVERT;
                if(sum_float_valid&&calc_state==C_CONVERT&&!failed)begin
                    row_sum<=sum_float_data;calc_state<=C_RECIP_REQ;
                end
                if(calc_state==C_RECIP_REQ&&reciprocal_s_ready&&!failed)calc_state<=C_RECIP_WAIT;
                if(calc_state==C_RECIP_WAIT&&reciprocal_valid&&!failed)begin
                    reciprocal<=reciprocal_data;calc_state<=C_NORM_WAIT;
                end
                if(begin_norm)begin calc_state<=C_NORM;norm_written<=0;end
                if(norm_fire)begin
                    norm_written<=norm_written+1'b1;
                    if({1'b0,norm_tag}+11'd1==words)begin
                        output_pending<=1;pending_row<=calc_row;calc_row<=calc_row+1'b1;
                        calc_state<=calc_row+17'd1==rows?C_FINISH:C_WAIT;
                    end
                end

                if(fault&&!failed)begin
                    failed<=1;error<=1;done<=0;
                    if(fault_read)error_code<=4;
                    else if(fault_short||fault_overrun)error_code<=6;
                    else if(fault_write)error_code<=5;
                    else if(fault_numeric)error_code<=2;
                    else error_code<=3;
                end
                if(failed)begin
                    calc_state<=C_IDLE;output_pending<=0;
                    if(load_state==L_IDLE&&store_state==W_IDLE&&!rd_busy&&!wr_busy&&!rxq_valid)busy<=0;
                end else if(calc_state==C_FINISH&&stored_rows==rows&&
                    store_state==W_IDLE&&load_state==L_IDLE&&!rd_busy&&!wr_busy&&!rxq_valid)begin
                    busy<=0;done<=1;calc_state<=C_IDLE;
                end
            end
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk)if(rst_n&&busy&&!failed)begin
        if(norm_ren&&store_ren)$fatal(1,"Output read-port collision");
        if(exp_fire&&norm_fire)$fatal(1,"Output write-port collision");
        if(load_fire&&load_guard&&load_count>=input_safe)$fatal(1,"Input overwrite before safe read");
        if(exp_fire&&exp_output_guard&&{1'b0,exp_tag}>=store_safe)$fatal(1,"Output overwrite before safe read");
    end
`endif
endmodule
