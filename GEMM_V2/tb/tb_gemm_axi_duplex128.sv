`timescale 1ns/1ps
module tb_gemm_axi_duplex128 #(parameter integer BURST=17);
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg rd_cmd_valid=0,wr_cmd_valid=0;
    reg [31:0] rd_cmd_addr=0,rd_cmd_bytes=0,wr_cmd_addr=0,wr_cmd_bytes=0;
    wire rd_cmd_ready,wr_cmd_ready,rd_done,wr_done,rd_error,wr_error,rd_busy,wr_busy;
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
    gemm_axi_duplex128 #(.C_M_AXI_ID_WIDTH(2),.MAX_BURST(BURST)) dut(
        .clk(clk),.rst_n(rst_n),
        .rd_cmd_valid(rd_cmd_valid),.rd_cmd_ready(rd_cmd_ready),.rd_cmd_addr(rd_cmd_addr),.rd_cmd_bytes(rd_cmd_bytes),
        .rd_done(rd_done),.rd_error(rd_error),.rd_busy(rd_busy),
        .wr_cmd_valid(wr_cmd_valid),.wr_cmd_ready(wr_cmd_ready),.wr_cmd_addr(wr_cmd_addr),.wr_cmd_bytes(wr_cmd_bytes),
        .wr_done(wr_done),.wr_error(wr_error),.wr_busy(wr_busy),
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
    integer rd_error_mode=0,wr_error_mode=0,burst_count=0,aw_count=0,b_count=0;
    reg [31:0] random_state=32'hc56a79b1;
    integer overlap_cycles=0;
    integer w_before_aw=0;
    reg rd_active=0,wr_addr_valid=0,wr_data_last=0;
    reg [31:0] rd_addr=0,wr_addr=0,rd_expected_base=0,wr_expected_base=0;
    reg [127:0] wr_words[0:255];
    integer i;
    // Memory model deliberately stalls AW for long enough to permit W before AW.
    always @(negedge clk) begin
        m_axi_arready = rst_n && !rd_active && !m_axi_rvalid && random_state[2:0]!=0;
        m_axi_awready = rst_n && !wr_addr_valid && random_state[4:3]==0;
        m_axi_wready = rst_n && !wr_data_last && random_state[6:5]!=0;
        rx_ready = rst_n && random_state[8:7]!=0;
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle<=0; rd_active<=0; m_axi_rvalid<=0; m_axi_bvalid<=0;
            wr_addr_valid<=0; wr_data_last<=0; wr_index<=0;
        end else begin
            cycle<=cycle+1;
            random_state<={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
            if(rd_busy && wr_busy)overlap_cycles<=overlap_cycles+1;
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arlen>=BURST || ({1'b0,m_axi_araddr[11:0]}+((m_axi_arlen+1)*16)>4096))
                    $fatal(1,"Illegal read burst");
                if (m_axi_arsize!=4 || m_axi_arburst!=1 || m_axi_arid!=0)
                    $fatal(1,"Illegal read attributes");
                if (m_axi_araddr != rd_expected_base + rd_count*16) $fatal(1,"Read address gap");
                rd_count<=rd_count+m_axi_arlen+1;
                rd_active<=1; rd_addr<=m_axi_araddr; rd_index<=0; rd_beats<=m_axi_arlen+1;
                burst_count<=burst_count+1;
            end
            if (m_axi_rvalid) begin
                if (m_axi_rready) begin
                    m_axi_rvalid<=0;
                    rd_index<=rd_index+1;
                    if (m_axi_rlast) rd_active<=0;
                end
            end else if (rd_active && random_state[10:9]!=0) begin
                m_axi_rvalid<=1;
                m_axi_rdata<=pattern(rd_addr+rd_index*16);
                m_axi_rresp<=((rd_error_mode==1 && rd_index==1)?2'b10:2'b00);
                m_axi_rid<=((rd_error_mode==2 && rd_index==1)?2'b01:2'b00);
                if (rd_error_mode==3) m_axi_rlast<=(rd_index==1);
                else if (rd_error_mode==4) m_axi_rlast<=(rd_index==rd_beats+1);
                else if (rd_error_mode==7) m_axi_rlast<=0;
                else m_axi_rlast<=(rd_index==rd_beats-1);
            end
            if (rx_valid && rx_ready) begin
                if (rx_data!==pattern(rd_expected_base+rx_count*16)) $fatal(1,"RX corrupt at %0d",rx_count);
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
                if (m_axi_awaddr != wr_expected_base + wr_count*16) $fatal(1,"Write address gap");
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
            if (wr_addr_valid && wr_data_last && !m_axi_bvalid && random_state[11]) begin
                if (wr_index!=wr_beats) $fatal(1,"WLAST or length incorrect");
                for (i=0;i<wr_beats;i=i+1)
                    if (wr_words[i]!==pattern(wr_addr+i*16)) $fatal(1,"TX corrupt");
                m_axi_bvalid<=1;
                m_axi_bid<=wr_error_mode==6?1:0;
                m_axi_bresp<=wr_error_mode==5?2'b10:0;
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
    task automatic issue_rd(input [31:0] addr,input [31:0] bytes,input integer mode);
        begin
            @(negedge clk);if(!rd_cmd_ready)$fatal(1,"RD not ready");
            rd_error_mode=mode;rd_expected_base=addr;rx_count=0;rd_count=0;burst_count=0;
            rd_cmd_valid=1;rd_cmd_addr=addr;rd_cmd_bytes=bytes;
            @(negedge clk);rd_cmd_valid=0;
        end
    endtask
    task automatic issue_wr(input [31:0] addr,input [31:0] bytes,input integer mode);
        begin
            @(negedge clk);if(!wr_cmd_ready)$fatal(1,"WR not ready");
            wr_error_mode=mode;wr_expected_base=addr;tx_count=0;wr_count=0;aw_count=0;b_count=0;
            wr_cmd_valid=1;wr_cmd_addr=addr;wr_cmd_bytes=bytes;
            @(negedge clk);wr_cmd_valid=0;
        end
    endtask
    task automatic wait_rd(input bit expected_error,input integer expected_rx);
        integer timeout;
        begin
            timeout=0;
            while(!rd_done && timeout<100000)begin @(negedge clk);timeout=timeout+1;end
            if(!rd_done || rd_error!==expected_error || rd_busy || rx_count!=expected_rx)
                $fatal(1,"RD completion error done=%b err=%b count=%0d expected=%0d",rd_done,rd_error,rx_count,expected_rx);
            @(negedge clk);if(rd_done || rd_error)$fatal(1,"RD completion not pulse");
        end
    endtask
    task automatic wait_wr(input bit expected_error,input integer expected_tx);
        integer timeout;
        begin
            timeout=0;
            while(!wr_done && timeout<100000)begin @(negedge clk);timeout=timeout+1;end
            if(!wr_done || wr_error!==expected_error || wr_busy || (expected_tx>=0&&tx_count!=expected_tx))
                $fatal(1,"WR completion error done=%b err=%b count=%0d expected=%0d",wr_done,wr_error,tx_count,expected_tx);
            @(negedge clk);if(wr_done || wr_error)$fatal(1,"WR completion not pulse");
        end
    endtask
    task automatic send_tx(input [31:0] a,input integer beats);
        integer t;
        begin
            for(t=0;t<beats;t=t+1)begin
                @(negedge clk);tx_valid=1;tx_data=pattern(a+t*16);
                @(posedge clk);while(!tx_ready)@(posedge clk);
                @(negedge clk);tx_valid=0;
            end
        end
    endtask
    task automatic send_error_tx(input [31:0] a);
        integer t;begin
            t=0;@(negedge clk);tx_valid=1;tx_data=pattern(a);
            while(!wr_done)begin
                @(posedge clk);if(tx_ready)t=t+1;
                @(negedge clk);tx_data=pattern(a+t*16);
            end
            tx_valid=0;
        end
    endtask
    initial begin
        repeat(5)@(negedge clk);rst_n=1;
        fork issue_rd(32'h0ff0,4320,0);issue_wr(32'h2ff0,4320,0);join
        fork send_tx(32'h2ff0,270);wait_rd(0,270);wait_wr(0,270);join
        if(overlap_cycles<100 || burst_count<3 || aw_count<3 || w_before_aw==0)
            $fatal(1,"Concurrency / splitting / AW-W independence not exercised");
        // Read failure must not prevent a much longer write from finishing.
        fork issue_rd(32'h4000,64,1);issue_wr(32'h6ff0,4320,0);join
        fork send_tx(32'h6ff0,270);wait_rd(1,1);wait_wr(0,270);join
        // Write failure must not prevent a much longer read from finishing.
        fork issue_rd(32'h8ff0,4320,0);issue_wr(32'ha000,1024,5);join
        fork send_error_tx(32'ha000);wait_rd(0,270);wait_wr(1,-1);join
        // Invalid command in one direction leaves the opposite direction available.
        fork issue_rd(32'h4001,32,0);issue_wr(32'hc000,64,0);join
        fork send_tx(32'hc000,4);wait_rd(1,0);wait_wr(0,4);join
        // Both directions are reusable after their independent errors.
        fork issue_rd(32'hd000,64,0);issue_wr(32'he000,64,0);join
        fork send_tx(32'he000,4);wait_rd(0,4);wait_wr(0,4);join
        $display("PASS tb_gemm_axi_duplex128 BURST=%0d: concurrent R/W, random backpressure, 4KB splits, isolated errors, recovery; overlap=%0d",BURST,overlap_cycles);
        $finish;
    end
    initial begin #3000000;$fatal(1,"Global timeout");end
endmodule
