`timescale 1ns/1ps
// ABI 10: full-K input strips (32 <= K < 16384), INT32 results, no partial-sum reload.
module gemm_core #(parameter integer C_USE_URAM=1)(
    input wire clk,rst_n,
    input wire [191:0] cfg_flat,input wire start_pulse,clear_status_pulse,
    output wire busy,irq,output reg [479:0] ro_flat,
    output wire dma_cmd_valid,input wire dma_cmd_ready,
    output wire [31:0] dma_cmd_addr,dma_cmd_bytes,
    input wire dma_done,dma_error,dma_busy,
    input wire [127:0] dma_rx_data,input wire dma_rx_valid,output wire dma_rx_ready,
    output wire [127:0] dma_tx_data,output wire dma_tx_valid,input wire dma_tx_ready,
    output wire wr_cmd_valid,input wire wr_cmd_ready,
    output wire [31:0] wr_cmd_addr,wr_cmd_bytes,wr_cmd_stride,output wire wr_cmd_strided,input wire wr_done,wr_error,wr_busy,
    input wire bus_read_beat,bus_write_beat
);
    localparam [4:0] IDLE=0,VALIDATE=1,AUTO_TILE=2,TILE_SETUP=3,CTX_SEND=4,
        CHECK_A=5,CHECK_BT=6,WAIT_A=7,WAIT_BT=8,SA_CLEAR=9,
        WORD_REQ=10,WORD_CAPTURE=11,FEED=12,DRAIN=13,C_STORE=14,
        WAIT_TILE=15,FINISH=16,FAIL=17,
        SPAN_PREP=18,SPAN_MUL=19,SPAN_CHECK=20,SPAN_RESULT=21,WORD_WAIT=22,TILE_JOIN=23,WAIT_WRITE=24,
        PLAN_DIV=25,PLAN_COST_INIT=26,PLAN_COST_MUL=27,PLAN_PICK=28;
    reg [4:0] state;
    // Declare shared nets/registers before their use in Verilog expressions.
    reg [8:0] tile_m,tile_n,mc,nc,mi,nj;
    wire sa_clear,sa_advance;
    localparam integer INPUT_READ_LATENCY=3;
    reg [1:0] input_wait;
    (* dont_touch="yes" *) reg config_bad_q;
    // Independent loader: the packer and SRAM writes keep running during
    // compute/writeback. Load alternating 32-row A/BT groups, schedule any ready pair.
    localparam [2:0] LD_IDLE=0,LD_CMD=1,LD_WAIT=2,LD_PICK=3;
    reg [2:0] load_state;
    reg [7:0] a_group_ready,bt_group_ready;
    reg [3:0] a_load_group,bt_load_group;
    reg prefer_bt;
    reg [2:0] load_group;
    reg [13:0] a_load_word_base,bt_load_word_base,load_sram_base,commit_count;
    reg [31:0] a_load_ddr,bt_load_ddr;
    reg [63:0] pair_pending;
    wire [7:0] eligible_a[0:7];
    wire [7:0] eligible_bt;
    genvar eg;
    generate for(eg=0;eg<8;eg=eg+1)begin:g_eligible
        assign eligible_a[eg]=a_group_ready & pair_pending[eg*8+:8] & {8{bt_group_ready[eg]}};
        assign eligible_bt[eg]=|eligible_a[eg];
    end endgenerate
    function automatic [2:0] first8(input [7:0] v);
        begin
            casez(v)
                8'b???????1:first8=0;8'b??????10:first8=1;
                8'b?????100:first8=2;8'b????1000:first8=3;
                8'b???10000:first8=4;8'b??100000:first8=5;
                8'b?1000000:first8=6;default:first8=7;
            endcase
        end
    endfunction
    wire [2:0] pick_bt=first8(eligible_bt);
    wire [2:0] pick_a=first8(eligible_a[pick_bt]);
    wire [5:0] selected_pair={nj[7:5],mi[7:5]};
    wire [63:0] selected_mask=64'h1<<selected_pair;
    function automatic [13:0] group_base(input [2:0] g,input [13:0] k);
        group_base=(g[0]?k:14'd0)+(g[1]?(k<<1):14'd0)+(g[2]?(k<<2):14'd0);
    endfunction
    integer pa,pb;
    reg task_active,done_q,error_q,perf_sat_q,write_done_seen;
    reg [191:0] cfg;
    wire [31:0] a_base=cfg[32+:32],bt_base=cfg[64+:32],y_base=cfg[96+:32];
    wire [31:0] dim_m={11'd0,cfg[128+:16],5'd0};
    wire [31:0] dim_n={11'd0,cfg[144+:16],5'd0};
    wire [31:0] dim_k={11'd0,cfg[160+:16],5'd0};
    wire [31:0] y_stride=dim_n<<2;
    reg [13:0] tile_k;
    reg [31:0] m0,n0;
    // 宏块按蛇形遍历；启动时比较两种方向的精确读入量，优先复用较大的缓存条带。
    // 只有遍历方向改变的操作数会失效，另一侧完整条带保留在输入缓存中。
    reg snake_columns,snake_reverse;
    reg [31:0] m_last,n_last,a_step_bytes,bt_step_bytes;
    reg [3:0] plan_step,plan_m_rem,plan_n_rem;
    reg [15:0] plan_m_bits,plan_n_bits,plan_m_quot,plan_n_quot;
    wire [3:0] plan_m_trial={plan_m_rem[2:0],plan_m_bits[15]};
    wire [3:0] plan_n_trial={plan_n_rem[2:0],plan_n_bits[15]};
    wire [16:0] plan_m_stripes={1'b0,plan_m_quot}+{16'd0,(plan_m_rem!=0)};
    wire [16:0] plan_n_stripes={1'b0,plan_n_quot}+{16'd0,(plan_n_rem!=0)};
    wire [31:0] plan_m_tail=(plan_m_rem!=0)?{23'd0,plan_m_rem,5'd0}:{23'd0,tile_m};
    wire [31:0] plan_n_tail=(plan_n_rem!=0)?{23'd0,plan_n_rem,5'd0}:{23'd0,tile_n};
    reg plan_row_reverse,plan_col_reverse;
    reg [39:0] plan_row_cost,plan_col_cost,plan_row_addend,plan_col_addend;
    reg [15:0] plan_row_pairs,plan_col_pairs,plan_k_bits;
    reg [31:0] plan_a_step,plan_bt_step,plan_a_last,plan_bt_last;
    reg [31:0] plan_a_step_addend,plan_bt_step_addend,plan_a_last_addend,plan_bt_last_addend;
    reg [8:0] k_word;
    reg load_kind;
    reg [14:0] load_count;
    reg [13:0] load_write_addr;
    reg [31:0] load_bytes;
    reg [4:0] k_byte,sa_row;
    reg [6:0] drain_count;
    // Address generators advance only when their row/group is consumed.
    reg [31:0] a_strip_base,bt_strip_base,next_a_strip,next_bt_strip,load_ddr_addr;
    reg [13:0] read_a_base,read_bt_base;
    reg [8:0] tile_row_cap;
    reg [11:0] c_write_ptr,c_micro_base;
    reg [6:0] produced_words;
    reg c_chunk;
    // One shared 16-cycle shift/add engine checks all three DDR spans.
    reg [1:0] span_index;reg [3:0] span_step;
    reg [15:0] span_bits;
    reg [31:0] span_addend,span_product,span_base;
    reg [63:0] span_end_q;
    reg a_valid,bt_valid;
    // Base, stride (=K), and job are implicit in cfg; START invalidates both tags.
    reg [31:0] a_tag_row,b_tag_row;
    reg [8:0] a_tag_rows,b_tag_rows;
    wire a_hit=a_valid&&a_tag_row==m0&&a_tag_rows==mc;
    wire bt_hit=bt_valid&&b_tag_row==n0&&b_tag_rows==nc;
    reg [31:0] accepted_job,compute_done_job,write_done_job,completed_tiles;
    reg [63:0] cycles_q,cycles_snap;
    reg [2:0] cycles_carry_q;
    wire [63:0] cycles_inc={
        cycles_q[63:48]+{15'd0,cycles_carry_q[2]},
        cycles_q[47:32]+{15'd0,cycles_carry_q[1]},
        cycles_q[31:16]+{15'd0,cycles_carry_q[0]},
        cycles_q[15:0]+16'd1};
    reg [31:0] compute_q,compute_snap,load_q,load_snap,store_q,read_bytes_q,write_bytes_q;
    reg dma_done_q,dma_error_q;
    assign busy=task_active;
    assign irq=cfg_flat[1]&&(done_q||error_q);
    function automatic [31:0] min32(input [31:0] a,b);min32=(a<b)?a:b;endfunction
    function automatic [31:0] max_rows(input [31:0] k);
        begin
            if(k<=2048)max_rows=256;else if(k<=2336)max_rows=224;
            else if(k<=2720)max_rows=192;else if(k<=3264)max_rows=160;
            else if(k<=4096)max_rows=128;else if(k<=5440)max_rows=96;
            else if(k<=8192)max_rows=64;else max_rows=32;
        end
    endfunction
    function automatic [31:0] sat_add(input [31:0] a,b);
        reg [32:0] t;begin t={1'b0,a}+{1'b0,b};sat_add=t[32]?32'hffffffff:t[31:0];end
    endfunction
    // Check the public encoded fields at START, in parallel with cfg capture.
    // This FF cuts the cfg->global fault->SRAM/FIFO enable timing cone.
    wire start_config_bad=(cfg_flat[143:128]==0)||(cfg_flat[159:144]==0)||
        (cfg_flat[175:160]==0)||(|cfg_flat[175:169])||(|cfg_flat[191:176])||
        (|cfg_flat[36:32])||(|cfg_flat[68:64])||(|cfg_flat[100:96]);
    wire config_bad=config_bad_q;
    wire span_bad_q=span_end_q>64'h100000000;
    // The selected row capacity replaces two wide runtime products.
    // Full software dimensions are checked before narrowing tile registers.
    wire auto_bad=(tile_m<32)||(tile_n<32)||(tile_k<32)||tile_m>256||tile_n>256||
        tile_m>tile_row_cap||tile_n>tile_row_cap||(|tile_m[4:0])||(|tile_n[4:0])||(|tile_k[4:0]);
    wire snake_inner_end=snake_columns?
        (snake_reverse?(m0==0):(m0+mc>=dim_m)):
        (snake_reverse?(n0==0):(n0+nc>=dim_n));
    wire last_tile=snake_inner_end&&(snake_columns?(n0+nc>=dim_n):(m0+mc>=dim_m));
    wire [8:0] k_words=dim_k[13:5];
    wire [255:0] input_pack_data;
    wire input_pack_valid,input_pack_ready,input_pack_empty,pack_rx_ready;
    wire writer_fault,writer_busy;
    // Two payload stages per operand. Linear row-major SRAM writes need no
    // bank decoder/broadcast; reset only the stage-valid tokens.
    reg [1:0] ab_wr_valid,ab_root_valid;
    reg [255:0] ab_wr_data[0:1],ab_root_data[0:1];
    reg [13:0] ab_wr_addr[0:1],ab_root_addr[0:1];
    wire ab_writes_empty=(ab_wr_valid==0)&&(ab_root_valid==0);
    wire fault_now=writer_fault || (task_active&&dma_done&&dma_error) ||
        (state==VALIDATE&&config_bad)||(state==SPAN_RESULT&&span_bad_q)||(state==TILE_SETUP&&auto_bad)||
        (load_state==LD_WAIT&&dma_done_q&&(dma_error_q||
         (input_pack_empty&&ab_writes_empty&&load_count!=(load_bytes>>5))))||
        (state>PLAN_PICK)||(load_state>LD_PICK);
    wire abort_all=error_q||fault_now||(state==FAIL);
    wire input_flush=!task_active||abort_all;
    gemm_pack128_256 u_input_pack(.clk(clk),.rst_n(rst_n),.flush(input_flush),
        .s_data(dma_rx_data),.s_valid(dma_rx_valid&&!abort_all),.s_ready(pack_rx_ready),
        .m_data(input_pack_data),.m_valid(input_pack_valid),.m_ready(input_pack_ready),.empty(input_pack_empty));
    assign dma_rx_ready=abort_all?1'b1:pack_rx_ready;
    // Loading owns the A/BT write port: no DDR receive FIFO or capacity feedback.
    assign input_pack_ready=(load_state==LD_WAIT)&&!abort_all;
    wire input_write=input_pack_valid&&input_pack_ready;
    assign dma_cmd_valid=(load_state==LD_CMD)&&!abort_all;
    assign dma_cmd_addr=load_ddr_addr;
    assign dma_cmd_bytes=load_bytes;
    wire [13:0] input_addr=load_write_addr;
    wire [13:0] read_a_addr,read_bt_addr;
    wire ctx_ready,micro_done;
    // Collect all 32 rows of columns 0..15, then 16..31. The
    // wavefront exposes the next row one clock later: no first-row bubbles.
    wire drain_store=(state==DRAIN)&&(drain_count>=17);
    wire collect_window=(state==C_STORE)||drain_store;
    wire store_fire=collect_window&&!abort_all;
    wire c_req_valid;wire [11:0] c_req_addr;
    reg c_req_delay,c_rsp_valid;
    wire [511:0] c_rsp_data;
    wire [31:0] writer_active_job;
    wire writer_engine_live=writer_busy||wr_busy;
    // 64 circular C slots, each 64 words (4 KiB). The producer reserves one
    // free slot; only the writer's full W/B completion releases it. Absolute
    // coordinates preserve correct output positions for either snake direction.
    (* ram_style="distributed" *) reg [64:0] c_context[0:63];
    reg [5:0] c_head,c_tail;
    reg [6:0] c_pending;
    reg c_inflight,c_producer_live;
    wire writer_live=writer_engine_live||(c_pending!=0);
    wire writer_ctx_ready;
    assign ctx_ready=(c_pending<64);
    wire c_enqueue=(state==CTX_SEND)&&ctx_ready&&!abort_all;
    wire c_capture_done=store_fire&&(produced_words==63);
    wire c_retire=micro_done&&c_inflight&&!abort_all;
    wire c_bookkeeping_clear=(state==IDLE&&start_pulse&&!error_q)||
        (state==FAIL&&!dma_busy&&!writer_engine_live);
    wire [64:0] writer_context=c_context[c_head];
    wire [31:0] micro_m0=writer_context[63:32];
    wire [31:0] micro_n0=writer_context[31:0];
    // 单 AXI 的读写通道独立：C 上下文可立即交给 writer，由 produced_words
    // 水位控制实际读出，不再等待整个宏块的 A/B 加载，允许加载/计算/回写重叠。
    wire ab_load_complete=(load_state==LD_IDLE)&&a_valid&&bt_valid&&
        !dma_busy&&input_pack_empty&&ab_writes_empty;
    wire writer_ctx_valid=(c_pending!=0)&&!c_inflight&&!abort_all;
    wire writer_ctx_fire=writer_ctx_valid&&writer_ctx_ready;
    // Older contexts are fully captured. Only the current producer can have
    // a partial result; retain its watermark on abort while accepted W drains.
    wire [6:0] writer_produced_words=
        (c_producer_live&&c_head==c_micro_base[11:6])?produced_words:7'd64;
    always @(posedge clk)if(c_enqueue)c_context[c_tail]<=
        {((pair_pending&~selected_mask)==0),(m0+{23'd0,mi}),(n0+{23'd0,nj})};
    always @(posedge clk)begin
        if(!rst_n||c_bookkeeping_clear)begin
            c_head<=0;c_tail<=0;c_pending<=0;c_inflight<=0;c_producer_live<=0;
        end else if(!abort_all)begin
            case({c_enqueue,c_retire})
                2'b10:c_pending<=c_pending+1'b1;
                2'b01:c_pending<=c_pending-1'b1;
                default:begin end
            endcase
            if(c_enqueue)begin c_tail<=c_tail+1'b1;c_producer_live<=1;end
            if(c_capture_done)c_producer_live<=0;
            if(writer_ctx_fire)c_inflight<=1;
            if(c_retire)begin c_head<=c_head+1'b1;c_inflight<=0;end
        end
    end
    gemm_output_writer512 #(.C_HALF_MAJOR(1)) u_writer(
        .clk(clk),.rst_n(rst_n),.cancel_all(abort_all),
        .ctx_valid(writer_ctx_valid),.ctx_ready(writer_ctx_ready),
        .ctx_job(accepted_job),.ctx_m0(micro_m0),.ctx_n0(micro_n0),
        .ctx_ybase(y_base),.ctx_ystride(y_stride),.ctx_cbase({c_head,6'd0}),
        .produced_words(writer_produced_words),
        .c_req_valid(c_req_valid),.c_req_addr(c_req_addr),.c_rsp_valid(c_rsp_valid),.c_rsp_data(c_rsp_data),
        .wr_cmd_valid(wr_cmd_valid),.wr_cmd_ready(wr_cmd_ready),.wr_cmd_addr(wr_cmd_addr),.wr_cmd_bytes(wr_cmd_bytes),.wr_cmd_strided(wr_cmd_strided),.wr_cmd_stride(wr_cmd_stride),
        .wr_done(wr_done),.wr_error(wr_error),.tx_data(dma_tx_data),.tx_valid(dma_tx_valid),.tx_ready(dma_tx_ready),
        .done(micro_done),.fault(writer_fault),.busy(writer_busy),.active_job(writer_active_job));
    always @(posedge clk)begin
        if(!rst_n)begin c_req_delay<=0;c_rsp_valid<=0;end
        else begin c_req_delay<=c_req_valid;c_rsp_valid<=c_req_delay;end
    end
    always @(posedge clk)begin
        if(input_write)begin
            ab_root_data[load_kind]<=input_pack_data;
            ab_root_addr[load_kind]<=input_addr;
        end
        if(!rst_n||abort_all)ab_root_valid<=0;
        else begin
            ab_root_valid<=0;
            if(input_write)ab_root_valid[load_kind]<=1;
        end
    end
    integer wg;
    always @(posedge clk)begin
        for(wg=0;wg<2;wg=wg+1)if(ab_root_valid[wg])begin
            ab_wr_data[wg]<=ab_root_data[wg];ab_wr_addr[wg]<=ab_root_addr[wg];
        end
        if(!rst_n||abort_all)ab_wr_valid<=0;
        else ab_wr_valid<=ab_root_valid;
    end
    wire a_we=ab_wr_valid[0],bt_we=ab_wr_valid[1];
    wire input_read_valid;
    wire [255:0] a_rdata,bt_rdata,a_vec,bt_vec;
    wire [31:0] a_lane_valid,bt_lane_valid;
    wire [1023:0] sa_result;
    wire [4:0] sa_read_row=collect_window?sa_row+5'd1:sa_row;
    // A/B 各为 512 KiB 逻辑缓存：各自前半 BRAM、后半 URAM，统一三级读延迟。
    // 两侧缓存都允许后续组写入与已就绪组计算并行。
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1),.MIXED_RAM(1)) u_a(
        .clk(clk),.we(a_we),.ren(input_read_valid),
        .waddr(ab_wr_addr[0]),.raddr(read_a_addr),.wdata(ab_wr_data[0]),.rdata(a_rdata));
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1),.MIXED_RAM(1)) u_bt(
        .clk(clk),.we(bt_we),.ren(input_read_valid),
        .waddr(ab_wr_addr[1]),.raddr(read_bt_addr),.wdata(ab_wr_data[1]),.rdata(bt_rdata));
    gemm_input_stream256 #(.READ_LATENCY(INPUT_READ_LATENCY)) u_input_stream(
        .clk(clk),.rst_n(rst_n),.clear(sa_clear),.cancel(abort_all),
        .start(state==WORD_REQ&&!abort_all),.advance(sa_advance),.k_words(k_words),
        .a_base(read_a_base),.bt_base(read_bt_base),.read_valid(input_read_valid),
        .a_addr(read_a_addr),.bt_addr(read_bt_addr),.a_data(a_rdata),.bt_data(bt_rdata),
        .a_vec(a_vec),.bt_vec(bt_vec),.a_valid(a_lane_valid),.bt_valid(bt_lane_valid));
    gemm_result_sram512 #(.RAM_STYLE(C_USE_URAM?"ultra":"block")) u_c(
        .clk(clk),.we(store_fire),.waddr(c_write_ptr),
        .wdata(sa_result[c_chunk*512+:512]),
        .ren(c_req_valid),.raddr(c_req_addr),.rdata(c_rsp_data));
    wire last_k_word=({1'b0,k_word}+10'd1)>={1'b0,k_words};
    // Predict next cycle's phase. Only local control FFs see abort_all;
    // DSP/data enables use their registered outputs, not the error cone.
    wire sa_advance_next=((state==WORD_CAPTURE)||
        (state==FEED)||
        (state==DRAIN&&drain_count!=66))&&!abort_all;
    wire sa_valid_next=((state==WORD_CAPTURE)||(state==FEED&&!(k_byte==31&&last_k_word)))&&!abort_all;
    // Preserve the original performance-counter definition on a failed cycle.
    // FIFO consumption follows the same already-clocked phase as the PEs.
    // A fault can discard this last registered step; no result may commit.
    assign sa_advance=(state==FEED||state==DRAIN);
    // Clear only at safe lifecycle boundaries. WAIT_TILE follows the last
    // physical C write, so DDR no longer needs to retain PE accumulators.
    // These clears remain safe if a fault coincides with the boundary.
    // Do not qualify them with abort_all: that routes FIFO length/error
    // checks into the high-fanout reset of every PE accumulator.
    assign sa_clear=(state==SA_CLEAR)||(state==WAIT_TILE)||
        (state==FAIL&&!dma_busy&&!writer_engine_live);
    gemm_systolic_array #(.N(32),.RESULT_PIPELINE(1),.LOCAL_CONTROL(1),.PRE_SKEWED(1)) u_array(.clk(clk),.rst_n(rst_n),.clear(sa_clear),.advance(sa_advance),
        .advance_next(sa_advance_next),.in_valid_next(sa_valid_next),
        .in_valid(state==FEED&&!abort_all),.a_vec(a_vec),.b_vec(bt_vec),
        .a_lane_valid(a_lane_valid),.b_lane_valid(bt_lane_valid),
        .result_row(sa_read_row),.result_data(sa_result));
    always @*begin
        ro_flat=0;
        ro_flat[0+:32]={6'd0,1'b0,tile_k[13:5],tile_n[8:5],tile_m[8:5],
            (write_done_seen&&write_done_job==accepted_job&&!error_q),writer_live,1'b0,perf_sat_q,1'b0,error_q,done_q,busy};
        ro_flat[32+:32]=completed_tiles;ro_flat[64+:32]=accepted_job;
        ro_flat[96+:32]=compute_done_job;ro_flat[128+:32]=write_done_job;
        ro_flat[160+:32]=writer_active_job;ro_flat[192+:32]=0;
        ro_flat[224+:64]=cycles_snap;ro_flat[288+:32]=compute_snap;ro_flat[320+:32]=load_snap;
        ro_flat[352+:32]=store_q;ro_flat[384+:32]=read_bytes_q;ro_flat[416+:32]=write_bytes_q;
        ro_flat[448+:32]=32'h0a080020;
    end
    always @(posedge clk)begin
        if(!rst_n)begin dma_done_q<=0;dma_error_q<=0;end
        else if(dma_cmd_valid&&dma_cmd_ready)begin dma_done_q<=0;dma_error_q<=0;end
        else if(dma_done)begin dma_done_q<=1;dma_error_q<=dma_error;end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            cfg<=0;config_bad_q<=0;input_wait<=0;state<=IDLE;load_state<=LD_IDLE;a_group_ready<=0;bt_group_ready<=0;
            task_active<=0;done_q<=0;error_q<=0;perf_sat_q<=0;write_done_seen<=0;
            tile_m<=0;tile_n<=0;tile_k<=0;m0<=0;n0<=0;mc<=0;nc<=0;mi<=0;nj<=0;
            snake_columns<=0;snake_reverse<=0;m_last<=0;n_last<=0;a_step_bytes<=0;bt_step_bytes<=0;
            plan_step<=0;plan_m_rem<=0;plan_n_rem<=0;plan_m_bits<=0;plan_n_bits<=0;
            plan_m_quot<=0;plan_n_quot<=0;plan_row_reverse<=0;plan_col_reverse<=0;
            plan_row_cost<=0;plan_col_cost<=0;plan_row_addend<=0;plan_col_addend<=0;
            plan_row_pairs<=0;plan_col_pairs<=0;plan_k_bits<=0;
            plan_a_step<=0;plan_bt_step<=0;plan_a_last<=0;plan_bt_last<=0;
            plan_a_step_addend<=0;plan_bt_step_addend<=0;plan_a_last_addend<=0;plan_bt_last_addend<=0;
            load_kind<=0;load_count<=0;load_write_addr<=0;load_bytes<=0;
            a_load_group<=0;bt_load_group<=0;prefer_bt<=0;load_group<=0;
            a_load_word_base<=0;bt_load_word_base<=0;load_sram_base<=0;commit_count<=0;
            a_load_ddr<=0;bt_load_ddr<=0;pair_pending<=0;
            k_word<=0;k_byte<=0;sa_row<=0;drain_count<=0;
            a_strip_base<=0;bt_strip_base<=0;next_a_strip<=0;next_bt_strip<=0;load_ddr_addr<=0;
            read_a_base<=0;read_bt_base<=0;tile_row_cap<=0;
            c_write_ptr<=0;c_micro_base<=0;produced_words<=0;c_chunk<=0;
            span_index<=0;span_step<=0;span_bits<=0;span_addend<=0;span_product<=0;span_base<=0;span_end_q<=0;
            a_valid<=0;bt_valid<=0;a_tag_row<=0;a_tag_rows<=0;
            b_tag_row<=0;b_tag_rows<=0;
            accepted_job<=0;compute_done_job<=0;write_done_job<=0;completed_tiles<=0;
            cycles_q<=0;cycles_carry_q<=0;cycles_snap<=0;compute_q<=0;compute_snap<=0;
            load_q<=0;load_snap<=0;store_q<=0;read_bytes_q<=0;write_bytes_q<=0;
        end else begin
            if(store_fire)begin
                c_write_ptr<=c_write_ptr+1'b1;
                produced_words<=produced_words+1'b1;
                sa_row<=sa_row+1'b1;
                if(sa_row==31)c_chunk<=!c_chunk;
            end
            if(c_enqueue)pair_pending<=pair_pending&~selected_mask;
            if(c_retire&&writer_context[64])completed_tiles<=completed_tiles+1'b1;
            if(clear_status_pulse)begin
                done_q<=0;perf_sat_q<=0;write_done_seen<=0;store_q<=0;read_bytes_q<=0;write_bytes_q<=0;
                if(!task_active&&!writer_live&&!dma_busy)error_q<=0;
            end
            if(task_active)begin
                cycles_q<=cycles_inc;
                // Look ahead to the all-ones condition of the next count.
                cycles_carry_q[0]<=(&cycles_q[15:1])&&!cycles_q[0];
                cycles_carry_q[1]<=(&cycles_q[31:1])&&!cycles_q[0];
                cycles_carry_q[2]<=(&cycles_q[47:1])&&!cycles_q[0];
                if(sa_advance&&!abort_all)begin compute_q<=sat_add(compute_q,1);if(compute_q==32'hffffffff)perf_sat_q<=1;end
                if(load_state!=LD_IDLE)begin load_q<=sat_add(load_q,1);if(load_q==32'hffffffff)perf_sat_q<=1;end
            end
            if(writer_live)begin store_q<=sat_add(clear_status_pulse?0:store_q,1);if(!clear_status_pulse&&store_q==32'hffffffff)perf_sat_q<=1;end
            if(bus_read_beat)begin read_bytes_q<=sat_add(clear_status_pulse?0:read_bytes_q,16);if(!clear_status_pulse&&read_bytes_q>32'hffffffef)perf_sat_q<=1;end
            if(bus_write_beat)begin write_bytes_q<=sat_add(clear_status_pulse?0:write_bytes_q,16);if(!clear_status_pulse&&write_bytes_q>32'hffffffef)perf_sat_q<=1;end
            if(fault_now)begin
                error_q<=1;write_done_seen<=0;a_valid<=0;bt_valid<=0;
                a_group_ready<=0;bt_group_ready<=0;load_state<=LD_IDLE;
                task_active<=1;state<=FAIL;
            end else begin
                // Physical commit, not DMA arrival, publishes a ready group.
                if(a_we||bt_we)begin
                    commit_count<=commit_count+1'b1;
                    if(commit_count+1'b1==tile_k)begin
                        if(load_kind)bt_group_ready[load_group]<=1;
                        else a_group_ready[load_group]<=1;
                    end
                end
                case(load_state)
                    LD_PICK:begin
                        if(a_load_group>=mc[8:5]&&bt_load_group>=nc[8:5])load_state<=LD_IDLE;
                        else begin
                            load_kind<=(bt_load_group<nc[8:5])&&((a_load_group>=mc[8:5])||prefer_bt);
                            if((bt_load_group<nc[8:5])&&((a_load_group>=mc[8:5])||prefer_bt))begin
                                load_group<=bt_load_group[2:0];load_ddr_addr<=bt_load_ddr;load_sram_base<=bt_load_word_base;
                            end else begin
                                load_group<=a_load_group[2:0];load_ddr_addr<=a_load_ddr;load_sram_base<=a_load_word_base;
                            end
                            load_bytes<=dim_k<<5;commit_count<=0;load_state<=LD_CMD;
                        end
                    end
                    LD_CMD:if(dma_cmd_ready)begin
                        load_count<=0;load_write_addr<=load_sram_base;load_state<=LD_WAIT;
                    end
                    LD_WAIT:begin
                        if(input_write)begin load_count<=load_count+1'b1;load_write_addr<=load_write_addr+1'b1;end
                        if(dma_done_q&&input_pack_empty&&ab_writes_empty)begin
                            prefer_bt<=!load_kind;load_state<=LD_PICK;
                            if(!load_kind)begin
                                a_load_group<=a_load_group+1'b1;
                                a_load_ddr<=load_ddr_addr+load_bytes;a_load_word_base<=a_load_word_base+tile_k;
                                if(a_load_group+1'b1>=mc[8:5])begin a_valid<=1;next_a_strip<=load_ddr_addr+load_bytes;end
                            end else begin
                                bt_load_group<=bt_load_group+1'b1;
                                bt_load_ddr<=load_ddr_addr+load_bytes;bt_load_word_base<=bt_load_word_base+tile_k;
                                if(bt_load_group+1'b1>=nc[8:5])begin bt_valid<=1;next_bt_strip<=load_ddr_addr+load_bytes;end
                            end
                        end
                    end
                    default:begin end
                endcase
                case(state)
                IDLE:if(start_pulse&&!error_q)begin
                    cfg<=cfg_flat;config_bad_q<=start_config_bad;task_active<=1;done_q<=0;perf_sat_q<=0;write_done_seen<=0;
                    accepted_job<=accepted_job+1'b1;completed_tiles<=0;
                    tile_m<=0;tile_n<=0;tile_k<=0;a_valid<=0;bt_valid<=0;
                    load_state<=LD_IDLE;a_group_ready<=0;bt_group_ready<=0;
                    cycles_q<=0;cycles_carry_q<=0;cycles_snap<=0;compute_q<=0;compute_snap<=0;load_q<=0;load_snap<=0;
                    state<=VALIDATE;
                end
                VALIDATE:begin span_index<=0;state<=SPAN_PREP;end
                SPAN_PREP:begin
                    span_step<=0;span_product<=0;
                    span_bits<=(span_index==1)?cfg[144+:16]:cfg[128+:16];
                    span_addend<={16'd0,(span_index==2)?cfg[144+:16]:cfg[160+:16]};
                    span_base<=(span_index==0)?a_base:(span_index==1)?bt_base:y_base;
                    state<=SPAN_MUL;
                end
                SPAN_MUL:begin
                    if(span_bits[0])span_product<=span_product+span_addend;
                    span_bits<=span_bits>>1;span_addend<=span_addend<<1;
                    if(span_step==15)state<=SPAN_CHECK;else span_step<=span_step+1'b1;
                end
                SPAN_CHECK:begin
                    // Dimension fields are in units of 32: input bytes <<10,
                    // output INT32 bytes <<12. No wide combinational multiplier.
                    span_end_q<={32'd0,span_base}+((span_index==2)?
                        {20'd0,span_product,12'd0}:{22'd0,span_product,10'd0});
                    state<=SPAN_RESULT;
                end
                SPAN_RESULT:if(span_index==2)begin tile_k<=dim_k;state<=AUTO_TILE;end
                    else begin span_index<=span_index+1'b1;state<=SPAN_PREP;end
                AUTO_TILE:begin
                    tile_m<=min32(dim_m,max_rows(tile_k));tile_n<=min32(dim_n,max_rows(tile_k));
                    tile_row_cap<=max_rows(tile_k);
                    // 单宏块没有方向选择，直接进入计算，保留小矩阵原有启动开销。
                    if(dim_m<=max_rows(tile_k)&&dim_n<=max_rows(tile_k))begin
                        m0<=0;n0<=0;m_last<=0;n_last<=0;snake_columns<=0;snake_reverse<=0;
                        a_strip_base<=a_base;bt_strip_base<=bt_base;a_step_bytes<=0;bt_step_bytes<=0;
                        state<=TILE_SETUP;
                    end else begin
                        // 16 拍恢复除法计算条带数及尾块，避免可变除法落入组合关键路径。
                        plan_m_bits<=cfg[128+:16];plan_n_bits<=cfg[144+:16];
                        plan_m_rem<=0;plan_n_rem<=0;plan_m_quot<=0;plan_n_quot<=0;
                        plan_step<=0;state<=PLAN_DIV;
                    end
                end
                PLAN_DIV:begin
                    plan_m_bits<=plan_m_bits<<1;plan_n_bits<=plan_n_bits<<1;
                    if(plan_m_trial>=tile_m[8:5])begin
                        plan_m_rem<=plan_m_trial-tile_m[8:5];
                        plan_m_quot<={plan_m_quot[14:0],1'b1};
                    end else begin plan_m_rem<=plan_m_trial;plan_m_quot<={plan_m_quot[14:0],1'b0};end
                    if(plan_n_trial>=tile_n[8:5])begin
                        plan_n_rem<=plan_n_trial-tile_n[8:5];
                        plan_n_quot<={plan_n_quot[14:0],1'b1};
                    end else begin plan_n_rem<=plan_n_trial;plan_n_quot<={plan_n_quot[14:0],1'b0};end
                    if(plan_step==15)state<=PLAN_COST_INIT;else plan_step<=plan_step+1'b1;
                end
                PLAN_COST_INIT:begin
                    m_last<=dim_m-plan_m_tail;n_last<=dim_n-plan_n_tail;
                    // 读入量以行计数，公共 K 因子无需参与比较；40 位覆盖最大合法维度。
                    // 每两次转弯各复用一个完整端点和一个尾块端点。
                    plan_row_pairs<=(plan_m_stripes-17'd1)>>1;
                    plan_col_pairs<=(plan_n_stripes-17'd1)>>1;
                    plan_row_addend<=({8'd0,dim_n}<<1)-{31'd0,tile_n}-{8'd0,plan_n_tail};
                    plan_col_addend<=({8'd0,dim_m}<<1)-{31'd0,tile_m}-{8'd0,plan_m_tail};
                    plan_row_cost<={8'd0,dim_m}+{8'd0,dim_n}+
                        (plan_m_stripes[0]?40'd0:({8'd0,dim_n}-{31'd0,tile_n}));
                    plan_col_cost<={8'd0,dim_m}+{8'd0,dim_n}+
                        (plan_n_stripes[0]?40'd0:({8'd0,dim_m}-{31'd0,tile_m}));
                    // 偶数条带时，先从较小尾块开始，能多保留一次较大的完整端点。
                    plan_row_reverse<=!plan_m_stripes[0]&&(plan_n_tail<{23'd0,tile_n});
                    plan_col_reverse<=!plan_n_stripes[0]&&(plan_m_tail<{23'd0,tile_m});
                    plan_a_step<=0;plan_bt_step<=0;plan_a_last<=0;plan_bt_last<=0;
                    plan_a_step_addend<={23'd0,tile_m};plan_bt_step_addend<={23'd0,tile_n};
                    plan_a_last_addend<=dim_m-plan_m_tail;plan_bt_last_addend<=dim_n-plan_n_tail;
                    plan_k_bits<=dim_k[15:0];plan_step<=0;state<=PLAN_COST_MUL;
                end
                PLAN_COST_MUL:begin
                    // 移位加法同时完成方向代价与地址步长，不引入宽组合乘法器。
                    if(plan_row_pairs[0])plan_row_cost<=plan_row_cost+plan_row_addend;
                    if(plan_col_pairs[0])plan_col_cost<=plan_col_cost+plan_col_addend;
                    plan_row_pairs<=plan_row_pairs>>1;plan_col_pairs<=plan_col_pairs>>1;
                    plan_row_addend<=plan_row_addend<<1;plan_col_addend<=plan_col_addend<<1;
                    if(plan_k_bits[0])begin
                        plan_a_step<=plan_a_step+plan_a_step_addend;
                        plan_bt_step<=plan_bt_step+plan_bt_step_addend;
                        plan_a_last<=plan_a_last+plan_a_last_addend;
                        plan_bt_last<=plan_bt_last+plan_bt_last_addend;
                    end
                    plan_k_bits<=plan_k_bits>>1;
                    plan_a_step_addend<=plan_a_step_addend<<1;plan_bt_step_addend<=plan_bt_step_addend<<1;
                    plan_a_last_addend<=plan_a_last_addend<<1;plan_bt_last_addend<=plan_bt_last_addend<<1;
                    if(plan_step==15)state<=PLAN_PICK;else plan_step<=plan_step+1'b1;
                end
                PLAN_PICK:begin
                    a_step_bytes<=plan_a_step;bt_step_bytes<=plan_bt_step;
                    if(plan_col_cost<plan_row_cost)begin
                        snake_columns<=1;snake_reverse<=plan_col_reverse;
                        m0<=plan_col_reverse?m_last:32'd0;n0<=0;
                        a_strip_base<=a_base+(plan_col_reverse?plan_a_last:32'd0);bt_strip_base<=bt_base;
                    end else begin
                        snake_columns<=0;snake_reverse<=plan_row_reverse;
                        m0<=0;n0<=plan_row_reverse?n_last:32'd0;
                        a_strip_base<=a_base;bt_strip_base<=bt_base+(plan_row_reverse?plan_bt_last:32'd0);
                    end
                    state<=TILE_SETUP;
                end
                TILE_SETUP:begin
                    mc<=min32(tile_m,dim_m-m0);nc<=min32(tile_n,dim_n-n0);mi<=0;nj<=0;
                    state<=CHECK_A;
                end
                CTX_SEND:if(ctx_ready)begin
                    c_write_ptr<={c_tail,6'd0};c_micro_base<={c_tail,6'd0};
                    produced_words<=0;state<=SA_CLEAR;
                end
                CHECK_A:begin
                    // Full-cache tags survive macroblocks; partial group maps
                    // belong only to this macroblock and are rebuilt on a miss.
                    if(!a_hit)begin a_valid<=0;a_group_ready<=0;end
                    if(!bt_hit)begin bt_valid<=0;bt_group_ready<=0;end
                    a_tag_row<=m0;a_tag_rows<=mc;b_tag_row<=n0;b_tag_rows<=nc;
                    a_load_group<=a_hit?mc[8:5]:4'd0;bt_load_group<=bt_hit?nc[8:5]:4'd0;
                    a_load_ddr<=a_strip_base;bt_load_ddr<=bt_strip_base;
                    a_load_word_base<=0;bt_load_word_base<=0;prefer_bt<=0;load_state<=LD_PICK;
                    for(pb=0;pb<8;pb=pb+1)for(pa=0;pa<8;pa=pa+1)
                        pair_pending[pb*8+pa]<=(pa<mc[8:5])&&(pb<nc[8:5]);
                    state<=WAIT_BT;
                end
                WAIT_BT:if(|eligible_bt)begin
                    mi<={1'b0,pick_a,5'd0};nj<={1'b0,pick_bt,5'd0};
                    read_a_base<=group_base(pick_a,tile_k);read_bt_base<=group_base(pick_bt,tile_k);
                    state<=CTX_SEND;
                end
                SA_CLEAR:begin
                    sa_row<=0;c_chunk<=0;k_word<=0;k_byte<=0;input_wait<=0;state<=WORD_REQ;
                end
                WORD_REQ:state<=WORD_WAIT;
                WORD_WAIT:if(input_wait==INPUT_READ_LATENCY-1)state<=WORD_CAPTURE;
                    else input_wait<=input_wait+1'b1;
                WORD_CAPTURE:begin
                    k_byte<=0;state<=FEED;
                end
                FEED:begin
                    if(k_byte==31)begin
                    k_byte<=0;
                    if(!last_k_word)begin k_word<=k_word+1'b1;state<=FEED;end
                    else begin drain_count<=0;state<=DRAIN;end
                end else k_byte<=k_byte+1'b1;
                end
                DRAIN:if(drain_count==66)state<=C_STORE;
                    else drain_count<=drain_count+1'b1;
                C_STORE:if(produced_words==63)state<=WAIT_TILE;
                WAIT_TILE:begin
                    // All 64 C words committed; DDR drains the reserved slot
                    // independently while the next ready A/BT pair computes.
                    produced_words<=0;
                    if(pair_pending==0)state<=TILE_JOIN;else state<=WAIT_BT;
                end
                // 当前计算已结束，只需确认 A/B 写入流水排空，即可覆盖失效的输入缓存。
                // C 以绝对坐标保留在环形队列，继续独立写回；只有整任务结束才等全部 W/B。
                TILE_JOIN:if(ab_load_complete)begin
                        if(last_tile)state<=WAIT_WRITE;
                        else begin
                            if(snake_columns)begin
                                if(!snake_inner_end)begin
                                    m0<=snake_reverse?(m0-{23'd0,tile_m}):(m0+{23'd0,tile_m});
                                    a_strip_base<=snake_reverse?(a_strip_base-a_step_bytes):(a_strip_base+a_step_bytes);
                                end else begin
                                    n0<=n0+{23'd0,nc};bt_strip_base<=bt_strip_base+bt_step_bytes;
                                    snake_reverse<=!snake_reverse;
                                end
                            end else begin
                                if(!snake_inner_end)begin
                                    n0<=snake_reverse?(n0-{23'd0,tile_n}):(n0+{23'd0,tile_n});
                                    bt_strip_base<=snake_reverse?(bt_strip_base-bt_step_bytes):(bt_strip_base+bt_step_bytes);
                                end else begin
                                    m0<=m0+{23'd0,mc};a_strip_base<=a_strip_base+a_step_bytes;
                                    snake_reverse<=!snake_reverse;
                                end
                            end
                            state<=TILE_SETUP;
                        end
                end
                WAIT_WRITE:if(c_pending==0&&!writer_engine_live)state<=FINISH;
                FINISH:begin task_active<=0;done_q<=1;compute_done_job<=accepted_job;
                    write_done_seen<=1;write_done_job<=accepted_job;produced_words<=0;c_write_ptr<=0;c_micro_base<=0;sa_row<=0;c_chunk<=0;
                    cycles_snap<=cycles_inc;compute_snap<=compute_q;load_snap<=load_q;state<=IDLE;end
                FAIL:if(!dma_busy&&!writer_engine_live)begin
                    produced_words<=0;c_write_ptr<=0;c_micro_base<=0;sa_row<=0;c_chunk<=0;
                    task_active<=0;cycles_snap<=cycles_inc;compute_snap<=compute_q;load_snap<=load_q;state<=IDLE;
                end
                default:begin end
            endcase
            end
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk)if(rst_n&&!abort_all)begin
        if(c_enqueue&&(!a_group_ready[mi[7:5]]||!bt_group_ready[nj[7:5]]||!pair_pending[selected_pair]))
            $fatal(1,"unready or repeated A/BT pair reserved");
        if(input_read_valid&&a_we&&read_a_addr==ab_wr_addr[0])
            $fatal(1,"A SRAM same-address read/write");
        if(input_read_valid&&bt_we&&read_bt_addr==ab_wr_addr[1])
            $fatal(1,"BT SRAM same-address read/write");
        if(c_pending>64||(c_retire&&c_pending==0))
            $fatal(1,"C context ownership count overflow/underflow");
        if(store_fire&&!c_producer_live)
            $fatal(1,"C write without a reserved producer region");
        if(writer_ctx_fire&&c_pending==0)
            $fatal(1,"writer accepted an unreserved C slot");
        if(c_enqueue&&c_producer_live)
            $fatal(1,"C producer reserved another slot before previous capture completed");
        if(c_inflight&&c_pending==0)
            $fatal(1,"C writer owns an empty context queue");
    end
`endif
endmodule
