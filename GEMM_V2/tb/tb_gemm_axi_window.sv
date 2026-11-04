`timescale 1ns/1ps
module tb_gemm_axi_window #(parameter integer BURST=17);
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg cmd_valid=0,cmd_write=0,cmd_strided=0;reg [31:0] cmd_stride=0;
    reg [31:0] cmd_addr=0,cmd_bytes=0;
    wire cmd_ready,done,error,busy;
    wire [127:0] rx_data; wire rx_valid; reg rx_ready=0;
    reg [127:0] tx_data=0; reg tx_valid=0; wire tx_ready;
    wire [1:0] m_axi_awid,m_axi_arid;
    wire [31:0] m_axi_awaddr,m_axi_araddr;
    wire [7:0] m_axi_awlen,m_axi_arlen;
    wire [2:0] m_axi_awsize,m_axi_arsize;
    wire [1:0] m_axi_awburst,m_axi_arburst;
    wire m_axi_awvalid,m_axi_arvalid,m_axi_wvalid,m_axi_wlast,m_axi_bready,m_axi_rready;
    reg m_axi_awready=0,m_axi_arready=0,m_axi_wready=0;
    wire [127:0] m_axi_wdata; wire [15:0] m_axi_wstrb;
    reg [1:0] m_axi_bid=0,m_axi_bresp=0,m_axi_rid=0,m_axi_rresp=0;
    reg m_axi_bvalid=0,m_axi_rvalid=0,m_axi_rlast=0;
    reg [127:0] m_axi_rdata=0;
    gemm_axi_dma128 #(.C_M_AXI_ID_WIDTH(2),.MAX_BURST(BURST),.RX_ALWAYS_READY(1),.STRIDED_WRITE(1)) dut(
        .clk(clk),.rst_n(rst_n),.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),
        .cmd_write(cmd_write),.cmd_addr(cmd_addr),.cmd_bytes(cmd_bytes),.cmd_strided(cmd_strided),.cmd_stride(cmd_stride),
        .done(done),.error(error),.busy(busy),
        .rx_data(rx_data),.rx_valid(rx_valid),.rx_ready(rx_ready),
        .tx_data(tx_data),.tx_valid(tx_valid),.tx_ready(tx_ready),
        .m_axi_awid(m_axi_awid),.m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),.m_axi_bresp(m_axi_bresp),.m_axi_buser(1'b0),
        .m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),.m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_ruser(1'b0),
        .m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready)
    );

    function automatic [127:0] pattern(input [31:0] a);
        pattern={a^32'h1234abcd,a^32'hf35a9876,a^32'h13579ace,a};
    endfunction
    integer cycle=0,ar_tail=0,r_head=0,aw_tail=0,w_head=0,b_head=0;
    integer ri=0,wi=0,rx_count=0,tx_count=0,ar_beats=0,aw_beats=0;
    integer max_rd=0,max_wr=0,mode=0,seen_fault=0,fault_cycle=0;
    integer read_latency=0,b_latency=0,overlap_w=0;
    integer r_streak=0,max_r_streak=0;
    reg [31:0] ra[0:1023],wa[0:1023];
    integer rn[0:1023],wn[0:1023];
    reg hold_second=0;
    always @(negedge clk)begin
        m_axi_arready=rst_n&&(ar_tail-r_head<4)&&(cycle%5!=0);
        m_axi_awready=rst_n&&(aw_tail-b_head<4)&&(cycle%7!=0);
        if(hold_second&&ar_tail==1&&(!seen_fault||cycle<fault_cycle+12))m_axi_arready=0;
        if(hold_second&&aw_tail==1&&(!seen_fault||cycle<fault_cycle+12))m_axi_awready=0;
        m_axi_wready=rst_n&&(w_head<aw_tail)&&(cycle%6!=0);
        rx_ready=rst_n;
        tx_valid=rst_n&&cmd_write&&busy;
        tx_data=pattern(cmd_addr+tx_count*16);
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            cycle<=0;ar_tail<=0;r_head<=0;aw_tail<=0;w_head<=0;b_head<=0;
            ri<=0;wi<=0;rx_count<=0;tx_count<=0;ar_beats<=0;aw_beats<=0;
            max_rd<=0;max_wr<=0;seen_fault<=0;fault_cycle<=0;
            read_latency<=0;b_latency<=0;overlap_w<=0;
            r_streak<=0;max_r_streak<=0;
            m_axi_rvalid<=0;m_axi_bvalid<=0;
        end else begin
            cycle<=cycle+1;
            if(ar_tail-r_head>max_rd)max_rd<=ar_tail-r_head;
            if(aw_tail-b_head>max_wr)max_wr<=aw_tail-b_head;
            if(m_axi_arvalid&&m_axi_arready)begin
                if(m_axi_arlen>=BURST||m_axi_araddr[11:0]+(m_axi_arlen+1)*16>4096||
                   m_axi_araddr!=cmd_addr+ar_beats*16)$fatal(1,"read descriptor/address");
                ra[ar_tail]<=m_axi_araddr;rn[ar_tail]<=m_axi_arlen+1;
                ar_tail<=ar_tail+1;ar_beats<=ar_beats+m_axi_arlen+1;
            end
            if(m_axi_rvalid&&m_axi_rready)begin
                m_axi_rvalid<=0;
                r_streak<=r_streak+1;
                if(r_streak+1>max_r_streak)max_r_streak<=r_streak+1;
                if(m_axi_rresp!=0||m_axi_rid!=0||m_axi_rlast!==(ri==rn[r_head]-1))begin
                    seen_fault<=1;if(!seen_fault)fault_cycle<=cycle;
                end
                if(m_axi_rlast)begin r_head<=r_head+1;ri<=0;end else ri<=ri+1;
                if(mode==0)begin
                    if(!m_axi_rlast)begin
                        m_axi_rvalid<=1;m_axi_rdata<=pattern(ra[r_head]+(ri+1)*16);
                        m_axi_rlast<=(ri+1==rn[r_head]-1);
                    end else if(r_head+1<ar_tail)begin
                        m_axi_rvalid<=1;m_axi_rdata<=pattern(ra[r_head+1]);m_axi_rlast<=(rn[r_head+1]==1);
                    end
                end
            end else if(!m_axi_rvalid&&r_head<ar_tail)begin
                // Real queued ARs: delay the first response so AR2 overlaps R1.
                if(read_latency<20)read_latency<=read_latency+1;
                else if(cycle%4!=0)begin
                    m_axi_rvalid<=1;m_axi_rid<=((mode==3&&r_head==0&&ri==0)?1:0);
                    m_axi_rresp<=((mode==1&&r_head==0&&ri==0)?2:0);
                    m_axi_rdata<=pattern(ra[r_head]+ri*16);
                    if(mode==4&&r_head==0&&BURST>1)m_axi_rlast<=1;
                    else if(mode==5&&r_head==0)m_axi_rlast<=(ri==rn[r_head]+1);
                    else m_axi_rlast<=(ri==rn[r_head]-1);
                end
            end
            if(rx_valid&&rx_ready)begin
                if(rx_data!==pattern(cmd_addr+rx_count*16))$fatal(1,"read stream mismatch");
                rx_count<=rx_count+1;
            end
            if(!(m_axi_rvalid&&m_axi_rready))r_streak<=0;
            if(m_axi_awvalid&&m_axi_awready)begin
                if(m_axi_awlen>=BURST||m_axi_awaddr[11:0]+(m_axi_awlen+1)*16>4096||
                   m_axi_awaddr!=(cmd_strided?cmd_addr+(aw_beats/8)*cmd_stride+(aw_beats%8)*16:cmd_addr+aw_beats*16))
                    $fatal(1,"write descriptor/stride/address");
                wa[aw_tail]<=m_axi_awaddr;wn[aw_tail]<=m_axi_awlen+1;
                aw_tail<=aw_tail+1;aw_beats<=aw_beats+m_axi_awlen+1;
            end
            if(m_axi_wvalid&&m_axi_wready)begin
                if(m_axi_wdata!==pattern(cmd_addr+tx_count*16)||m_axi_wstrb!=16'hffff||
                   m_axi_wlast!==(wi==wn[w_head]-1))$fatal(1,"W stream/last mismatch");
                tx_count<=tx_count+1;
                if(w_head>b_head)overlap_w<=overlap_w+1;
                if(m_axi_wlast)begin w_head<=w_head+1;wi<=0;end else wi<=wi+1;
            end
            if(m_axi_bvalid&&m_axi_bready)begin
                m_axi_bvalid<=0;b_head<=b_head+1;
                if(m_axi_bresp!=0||m_axi_bid!=0)begin seen_fault<=1;fault_cycle<=cycle;end
            end else if(!m_axi_bvalid&&b_head<w_head)begin
                if(b_latency<35)b_latency<=b_latency+1;
                else if(cycle%3!=0)begin
                m_axi_bvalid<=1;m_axi_bid<=((mode==6&&b_head==0)?1:0);
                m_axi_bresp<=((mode==2&&b_head==0)?2:0);
                end
            end
            if(m_axi_rvalid&&!m_axi_rready)$fatal(1,"read backpressure");
        end
    end
    reg ar_stall=0,aw_stall=0,w_stall=0;
    reg [39:0] ar_hold,aw_hold;reg [144:0] w_hold;
    always @(posedge clk)begin
        if(rst_n)begin
            if(ar_stall&&(!m_axi_arvalid||{m_axi_araddr,m_axi_arlen}!==ar_hold))$fatal(1,"held AR changed on fault");
            if(aw_stall&&(!m_axi_awvalid||{m_axi_awaddr,m_axi_awlen}!==aw_hold))$fatal(1,"held AW changed on fault");
            if(w_stall&&(!m_axi_wvalid||{m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==w_hold))$fatal(1,"held W changed");
        end
        ar_stall<=rst_n&&m_axi_arvalid&&!m_axi_arready;
        aw_stall<=rst_n&&m_axi_awvalid&&!m_axi_awready;
        w_stall<=rst_n&&m_axi_wvalid&&!m_axi_wready;
        ar_hold<={m_axi_araddr,m_axi_arlen};aw_hold<={m_axi_awaddr,m_axi_awlen};
        w_hold<={m_axi_wlast,m_axi_wstrb,m_axi_wdata};
    end
    task automatic run(input bit write_mode,stride_mode,input integer error_mode);
        integer timeout;begin
            @(negedge clk);rst_n=0;cmd_valid=0;
            repeat(3)@(negedge clk);
            cmd_write=write_mode;cmd_strided=stride_mode;cmd_stride=544;mode=error_mode;
            hold_second=(error_mode!=0);cmd_addr=(error_mode==4)?32'h4000:32'h00000ff0;cmd_bytes=stride_mode?4096:4320;
            rst_n=1;@(negedge clk);cmd_valid=1;@(negedge clk);cmd_valid=0;
            timeout=0;
            while(!done)begin @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"window timeout");end
            if(error!==(error_mode!=0)||busy||r_head!=ar_tail||b_head!=aw_tail||w_head!=aw_tail)
                $fatal(1,"premature completion or missing error");
            if(error_mode==0)begin
                if(write_mode)begin
                    if(tx_count!=cmd_bytes/16||max_wr!=2||overlap_w==0)$fatal(1,"write overlap not exercised");
                end else begin
                    if(rx_count!=cmd_bytes/16||max_rd!=2)$fatal(1,"read overlap not exercised");
                    if(BURST>=17&&max_r_streak<BURST+1)$fatal(1,"no continuous R across queued burst boundary");
                end
            end else if(!seen_fault||ar_tail>2||aw_tail>2)$fatal(1,"fault did not stop new allocation");
            $display("PASS window BURST=%0d write=%0d stride=%0d error=%0d ARpeak=%0d AWpeak=%0d WbeforeB=%0d Rstreak=%0d",BURST,write_mode,stride_mode,error_mode,max_rd,max_wr,overlap_w,max_r_streak);
            @(negedge clk);
        end
    endtask
    initial begin
        run(0,0,0);run(1,0,0);run(1,1,0);run(0,0,1);run(0,0,3);
        if(BURST>1)run(0,0,4);
        run(0,0,5);run(1,0,2);run(1,0,6);run(1,1,0);run(0,0,0);
        $display("PASS two-burst window, 4KiB/stride split, held address fault drain, recovery");$finish;
    end
    initial begin #10000000;$fatal(1,"window global timeout");end
endmodule
