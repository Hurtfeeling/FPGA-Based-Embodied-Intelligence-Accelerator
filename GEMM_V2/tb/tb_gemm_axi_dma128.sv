`timescale 1ns/1ps
module tb_gemm_axi_dma128 #(parameter integer BURST=17,NO_BACKPRESSURE=0,BW_ONLY=0);
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg cmd_valid=0,cmd_write=0;
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
    gemm_axi_dma128 #(.C_M_AXI_ID_WIDTH(2),.MAX_BURST(BURST),.RX_ALWAYS_READY(NO_BACKPRESSURE)) dut(
        .clk(clk),.rst_n(rst_n),.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),
        .cmd_write(cmd_write),.cmd_addr(cmd_addr),.cmd_bytes(cmd_bytes),
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
        pattern = {a ^ 32'hcafe0101,a ^ 32'h10203040,a ^ 32'hfedcba98,a};
    endfunction
    integer cycle=0,rd_count=0,rd_index=0,rd_beats=0;
    integer wr_count=0,wr_index=0,wr_beats=0,rx_count=0,tx_count=0;
    integer error_mode=0,burst_count=0,aw_count=0,b_count=0;
    integer w_before_aw=0;
    reg full_rate=0;
    integer r_streak=0,w_streak=0,max_r_streak=0,max_w_streak=0;
    reg rd_active=0,wr_addr_valid=0,wr_data_last=0;
    reg [31:0] rd_addr=0,wr_addr=0,expected_base=0;
    reg [127:0] wr_words[0:255];
    integer i;
    // Memory model deliberately stalls AW for long enough to permit W before AW.
    always @(negedge clk) begin
        m_axi_arready = rst_n && !rd_active && !m_axi_rvalid && cycle%5!=0;
        m_axi_awready = rst_n && !wr_addr_valid && cycle%13==0;
        m_axi_wready = rst_n && !wr_data_last && (full_rate||cycle%4!=0);
        rx_ready = rst_n && (full_rate||(cycle%7!=0 && cycle%7!=1));
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle<=0; rd_active<=0; m_axi_rvalid<=0; m_axi_bvalid<=0;
            wr_addr_valid<=0; wr_data_last<=0; wr_index<=0;
        end else begin
            cycle<=cycle+1;
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arlen>=BURST || ({1'b0,m_axi_araddr[11:0]}+((m_axi_arlen+1)*16)>4096))
                    $fatal(1,"Illegal read burst");
                if (m_axi_arsize!=4 || m_axi_arburst!=1 || m_axi_arid!=0)
                    $fatal(1,"Illegal read attributes");
                if (m_axi_araddr != expected_base + rd_count*16) $fatal(1,"Read address gap");
                rd_count<=rd_count+m_axi_arlen+1;
                rd_active<=1; rd_addr<=m_axi_araddr; rd_index<=0; rd_beats<=m_axi_arlen+1;
                burst_count<=burst_count+1;
            end
            if(full_rate)begin
                if(m_axi_rvalid&&m_axi_rready)begin
                    rd_index<=rd_index+1;
                    if(m_axi_rlast)begin m_axi_rvalid<=0;rd_active<=0;end
                    else begin
                        m_axi_rvalid<=1;m_axi_rdata<=pattern(rd_addr+(rd_index+1)*16);
                        m_axi_rlast<=(rd_index+1==rd_beats-1);
                    end
                end else if(!m_axi_rvalid&&rd_active)begin
                    m_axi_rvalid<=1;m_axi_rdata<=pattern(rd_addr+rd_index*16);
                    m_axi_rid<=0;m_axi_rresp<=0;m_axi_rlast<=(rd_index==rd_beats-1);
                end
            end else if (m_axi_rvalid) begin
                if (m_axi_rready) begin
                    m_axi_rvalid<=0;
                    rd_index<=rd_index+1;
                    if (m_axi_rlast) rd_active<=0;
                end
            end else if (rd_active && cycle%3!=0) begin
                m_axi_rvalid<=1;
                m_axi_rdata<=pattern(rd_addr+rd_index*16);
                m_axi_rresp<=((error_mode==1 && rd_index==1)?2'b10:2'b00);
                m_axi_rid<=((error_mode==2 && rd_index==1)?2'b01:2'b00);
                if (error_mode==3) m_axi_rlast<=(rd_index==1);
                else if (error_mode==4) m_axi_rlast<=(rd_index==rd_beats+1);
                else if (error_mode==7) m_axi_rlast<=0;
                else m_axi_rlast<=(rd_index==rd_beats-1);
            end
            if (rx_valid && rx_ready) begin
                if (rx_data!==pattern(expected_base+rx_count*16)) $fatal(1,"RX corrupt at %0d",rx_count);
                rx_count<=rx_count+1;
            end
            if (tx_valid && tx_ready) begin
                if (!(m_axi_wvalid && m_axi_wready)) $fatal(1,"TX handshake not a W handshake");
                tx_count<=tx_count+1;
            end
            if (m_axi_awvalid && m_axi_awready) begin
                if (m_axi_awlen>=BURST || ({1'b0,m_axi_awaddr[11:0]}+((m_axi_awlen+1)*16)>4096))
                    $fatal(1,"Illegal write burst");
                if (m_axi_awsize!=4 || m_axi_awburst!=1 || m_axi_awid!=0)
                    $fatal(1,"Illegal write attributes");
                if (m_axi_awaddr != expected_base + wr_count*16) $fatal(1,"Write address gap");
                wr_count<=wr_count+m_axi_awlen+1;
                wr_addr_valid<=1; wr_addr<=m_axi_awaddr; wr_beats<=m_axi_awlen+1;
                aw_count<=aw_count+1;
            end
            if (m_axi_wvalid && m_axi_wready) begin
                if (!wr_addr_valid) w_before_aw<=w_before_aw+1;
                if (m_axi_wstrb!=16'hffff) $fatal(1,"WSTRB");
                wr_words[wr_index]<=m_axi_wdata;
                wr_index<=wr_index+1;
                if (m_axi_wlast) wr_data_last<=1;
            end
            if (wr_addr_valid && wr_data_last && !m_axi_bvalid) begin
                if (wr_index!=wr_beats) $fatal(1,"WLAST or length incorrect");
                for (i=0;i<wr_beats;i=i+1)
                    if (wr_words[i]!==pattern(wr_addr+i*16)) $fatal(1,"TX corrupt");
                m_axi_bvalid<=1;
                m_axi_bid<=error_mode==6?1:0;
                m_axi_bresp<=error_mode==5?2'b10:0;
            end
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid<=0; wr_addr_valid<=0; wr_data_last<=0; wr_index<=0;
                b_count<=b_count+1;
            end
        end
    end
    reg stalled_aw=0,stalled_ar=0,stalled_w=0;
    reg [39:0] aw_hold,ar_hold; reg [144:0] w_hold;
    always @(posedge clk) begin
        if (rst_n) begin
            if(NO_BACKPRESSURE&&rd_active&&m_axi_rvalid&&!m_axi_rready)
                $fatal(1,"AXI reader inserted backpressure during an accepted burst");
            if(m_axi_rvalid&&m_axi_rready)begin
                r_streak=r_streak+1;if(r_streak>max_r_streak)max_r_streak=r_streak;
            end else r_streak=0;
            if(m_axi_wvalid&&m_axi_wready)begin
                w_streak=w_streak+1;if(w_streak>max_w_streak)max_w_streak=w_streak;
            end else w_streak=0;
            if (stalled_aw && (!m_axi_awvalid || {m_axi_awaddr,m_axi_awlen}!==aw_hold)) $fatal(1,"AW changed while stalled");
            if (stalled_ar && (!m_axi_arvalid || {m_axi_araddr,m_axi_arlen}!==ar_hold)) $fatal(1,"AR changed while stalled");
            if (stalled_w && (!m_axi_wvalid || {m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==w_hold)) $fatal(1,"W changed while stalled");
        end
        stalled_aw<=rst_n && m_axi_awvalid && !m_axi_awready;
        stalled_ar<=rst_n && m_axi_arvalid && !m_axi_arready;
        stalled_w<=rst_n && m_axi_wvalid && !m_axi_wready;
        aw_hold<={m_axi_awaddr,m_axi_awlen}; ar_hold<={m_axi_araddr,m_axi_arlen};
        w_hold<={m_axi_wlast,m_axi_wstrb,m_axi_wdata};
    end
    task automatic issue(input bit wr,input [31:0] addr,input [31:0] bytes,input integer mode);
        begin
            @(negedge clk);
            if (!cmd_ready) $fatal(1,"Not ready for command");
            error_mode=mode; expected_base=addr;
            rx_count=0; tx_count=0; rd_count=0; wr_count=0; burst_count=0; aw_count=0; b_count=0;
            r_streak=0;w_streak=0;max_r_streak=0;max_w_streak=0;
            cmd_valid=1; cmd_write=wr; cmd_addr=addr; cmd_bytes=bytes;
            @(negedge clk); cmd_valid=0;
        end
    endtask
    task automatic wait_done(input bit expected_error,input integer expected_rx,input integer expected_tx);
        integer timeout;
        begin
            timeout=0;
            while (!done && timeout<100000) begin @(negedge clk); timeout=timeout+1; end
            if (!done || error!==expected_error) $fatal(1,"Completion error expected %0d got done=%b err=%b",expected_error,done,error);
            if ((expected_rx>=0&&rx_count!=expected_rx) || (expected_tx>=0&&tx_count!=expected_tx))
                $fatal(1,"Count expected rx/tx=%0d/%0d got %0d/%0d",expected_rx,expected_tx,rx_count,tx_count);
            if (busy) $fatal(1,"BUSY at completion");
            @(negedge clk);
            if (done || error) $fatal(1,"Completion not a pulse");
        end
    endtask
    task automatic send_tx(input [31:0] a,input integer beats);
        integer t;
        begin
            for (t=0;t<beats;t=t+1) begin
                @(negedge clk); tx_valid=1;tx_data=pattern(a+t*16);
                @(posedge clk);
                while (!tx_ready) @(posedge clk);
                @(negedge clk);tx_valid=0;
            end
        end
    endtask
    task automatic send_error_tx(input [31:0] a);
        integer t;begin
            t=0;
            @(negedge clk);tx_valid=1;tx_data=pattern(a);
            while(!done)begin
                @(posedge clk);
                if(tx_ready)begin t=t+1;end
                @(negedge clk);tx_data=pattern(a+t*16);
            end
            tx_valid=0;
        end
    endtask
    task automatic send_tx_dense(input [31:0] a,input integer beats);
        integer t;
        begin
            @(negedge clk);tx_valid=1;tx_data=pattern(a);
            for(t=0;t<beats;t=t+1)begin
                @(posedge clk);while(!tx_ready)@(posedge clk);
                @(negedge clk);
                if(t==beats-1)tx_valid=0;else tx_data=pattern(a+(t+1)*16);
            end
        end
    endtask
    initial begin
        repeat(5) @(negedge clk);rst_n=1;
        if(!BW_ONLY)begin
        issue(0,32'h00000ff0,32'd4320,0);
        wait_done(0,270,0);
        if (burst_count<3) $fatal(1,"Read split not exercised");
        issue(1,32'h00002ff0,32'd4320,0);
        fork
            send_tx(32'h00002ff0,270);
            wait_done(0,0,270);
        join
        if (aw_count!=b_count || aw_count<3 || w_before_aw==0) $fatal(1,"Write responses / independent AW-W not exercised");
        issue(0,32'h4000,64,1);wait_done(1,1,0);
        if (burst_count!=1) $fatal(1,"Continued after RRESP error");
        issue(0,32'h4000,64,2);wait_done(1,1,0);
        issue(0,32'h4000,64,3);wait_done(1,1,0);
        issue(0,32'h4000,64,4);wait_done(1,3,0);
        issue(1,32'h5000,1024,5);
        fork send_error_tx(32'h5000);wait_done(1,0,-1);join
        if(aw_count!=b_count||aw_count>2)$fatal(1,"Failed write did not drain reserved bursts");
        issue(1,32'h5000,64,6);
        fork send_tx(32'h5000,4);wait_done(1,0,4);join
        issue(0,32'h4001,32,0);wait_done(1,0,0);
        issue(0,32'h4000,0,0);wait_done(1,0,0);
        issue(0,32'h4000,17,0);wait_done(1,0,0);
        issue(0,32'hfffffff0,32,0);wait_done(1,0,0);
        issue(0,32'hfffffff0,16,0);wait_done(0,1,0);
        issue(0,32'h4000,64,0);wait_done(0,4,0);
        // No observed RLAST must never create a synthetic completion.
        issue(0,32'h4000,64,7);
        repeat(60) begin @(negedge clk); if (done || !busy) $fatal(1,"Completed before RLAST"); end
        rst_n=0;repeat(3) @(negedge clk);rst_n=1;
        issue(0,32'h4000,64,0);wait_done(0,4,0);
        end
        full_rate=1;
        issue(0,32'h10000,4*BURST*16,0);wait_done(0,4*BURST,0);
        if(max_r_streak!=BURST||burst_count!=4)
            $fatal(1,"Read burst contains bubbles max_streak=%0d burst=%0d",max_r_streak,BURST);
        $display("PASS dense AXI R: BURST=%0d, %0d consecutive beats, 4 bursts",BURST,max_r_streak);
        issue(1,32'h20000,4*BURST*16,0);
        fork send_tx_dense(32'h20000,4*BURST);wait_done(0,0,4*BURST);join
        if(max_w_streak!=BURST||aw_count!=4)
            $fatal(1,"Write burst contains internal bubbles max_streak=%0d burst=%0d",max_w_streak,BURST);
        $display("PASS dense AXI W: BURST=%0d, %0d consecutive beats, 4 bursts",BURST,max_w_streak);
        $display("PASS tb_gemm_axi_dma128: 4KB/maxburst splits, backpressure, AW/W independence, R/B errors, RLAST drain, invalid commands, recovery");
        $finish;
    end
    initial begin #3000000;$fatal(1,"Global timeout");end
endmodule
