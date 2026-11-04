`timescale 1ns/1ps
// End-to-end CPU/DDR reference; hierarchy only observes ownership and error residue.
module tb_gemm_accelerator #(parameter integer CASE=0,parameter integer DMA_BURST=17,parameter integer C_USE_URAM=1,parameter integer DENSE_DDR=0);
    localparam [31:0] AXIL_BASE=32'h80000000;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] S_AXI_AWADDR=0,S_AXI_ARADDR=0;
    reg [2:0] S_AXI_AWPROT=0,S_AXI_ARPROT=0;
    reg S_AXI_AWVALID=0,S_AXI_WVALID=0,S_AXI_BREADY=0,S_AXI_ARVALID=0,S_AXI_RREADY=0;
    wire S_AXI_AWREADY,S_AXI_WREADY,S_AXI_BVALID,S_AXI_ARREADY,S_AXI_RVALID;
    reg [31:0] S_AXI_WDATA=0; reg [3:0] S_AXI_WSTRB=0;
    wire [31:0] S_AXI_RDATA; wire [1:0] S_AXI_BRESP,S_AXI_RRESP;
    wire [0:0] m_axi_awid,m_axi_arid;
    wire [31:0] m_axi_awaddr,m_axi_araddr;
    wire [7:0] m_axi_awlen,m_axi_arlen;
    wire [2:0] m_axi_awsize,m_axi_arsize;
    wire [1:0] m_axi_awburst,m_axi_arburst;
    wire m_axi_awlock,m_axi_arlock;
    wire [3:0] m_axi_awcache,m_axi_arcache,m_axi_awqos,m_axi_arqos;
    wire [2:0] m_axi_awprot,m_axi_arprot;
    wire m_axi_awvalid,m_axi_arvalid,m_axi_wvalid,m_axi_wlast,m_axi_bready,m_axi_rready;
    reg m_axi_awready=0,m_axi_arready=0,m_axi_wready=0;
    wire [127:0] m_axi_wdata; wire [15:0] m_axi_wstrb;
    reg [0:0] m_axi_bid=0,m_axi_rid=0;
    reg [1:0] m_axi_bresp=0,m_axi_rresp=0;
    reg m_axi_bvalid=0,m_axi_rvalid=0,m_axi_rlast=0;
    reg [127:0] m_axi_rdata=0;
    // Exercise the public default base; the AXI-Lite unit test also covers an override.
    gemm_accelerator_top #(.MAX_BURST(DMA_BURST),.C_USE_URAM(C_USE_URAM)) dut(.aclk(clk),.aresetn(rst_n),.*);

    localparam integer MEM_BYTES=8388608;
    localparam integer A_BASE=32'h10000,BT_BASE=32'h200000,Y_BASE=32'h400000;
    reg [7:0] memory [0:MEM_BYTES-1];
    reg [127:0] write_burst[0:255];
    reg [31:0] random_state=32'h4c734a91;
    integer cycles=0,read_bursts=0,write_bursts=0,read_beats=0,write_beats=0;
    integer r_streak=0,max_r_streak=0;
    integer rd_index=0,rd_beats=0,wr_index=0,wr_beats=0;
    reg rd_active=0,wr_addr_valid=0,wr_data_last=0;
    reg [31:0] rd_addr=0,wr_addr=0;
    integer b_delay=0;
    reg hold_write_response=0,hold_hit=0,monitor_overlap=0;
    reg inject_r_error=0,inject_b_error=0,inject_r_done=0,inject_b_done=0;
    reg [31:0] inject_r_addr=A_BASE;
    integer hold_bt_from=-1,hold_a_from=-1;
    function automatic bit held_read(input [31:0] addr);
        held_read=(hold_a_from>=0&&addr>=A_BASE+hold_a_from*test_k&&addr<A_BASE+test_m*test_k)||
                  (hold_bt_from>=0&&addr>=BT_BASE+hold_bt_from*test_k&&addr<BT_BASE+test_n*test_k);
    endfunction
    integer bg_error_phase=-1;
    reg hold_w_data=0;
    reg r_error_had_half=0,b_error_had_tail=0;
    reg [31:0] inject_b_addr=Y_BASE;
    wire write_error_target=inject_b_error && !inject_b_done && wr_addr==inject_b_addr;
    always @(posedge clk)begin
        if(rst_n)begin
            if(m_axi_rvalid&&!m_axi_rready)$fatal(1,"GEMM inserted AXI read backpressure");
            if(m_axi_rvalid&&m_axi_rready)begin
                r_streak=r_streak+1;if(r_streak>max_r_streak)max_r_streak=r_streak;
            end else r_streak=0;
        end
        if(rst_n && m_axi_rvalid && m_axi_rready && m_axi_rresp!=0 && inject_r_error)begin
            inject_r_done<=1;r_error_had_half<=!dut.u_core.input_pack_empty;
            if(dut.u_core.input_pack_empty)$fatal(1,"Read error did not leave an unmatched low 128-bit half");
            if(bg_error_phase==1&&!(dut.u_core.state==dut.u_core.FEED||dut.u_core.state==dut.u_core.WORD_REQ||
                dut.u_core.state==dut.u_core.WORD_WAIT||dut.u_core.state==dut.u_core.WORD_CAPTURE))
                $fatal(1,"background read error did not overlap first computation");
        end
        if(rst_n && m_axi_bvalid && m_axi_bready && m_axi_bresp!=0 && inject_b_error)begin
            inject_b_done<=1;b_error_had_tail<=!dut.u_core.u_writer.stream_empty;
            if(wr_beats!=((test_n==32)?DMA_BURST:8) || dut.u_core.u_writer.stream_empty)
                $fatal(1,"Write error did not leave a producer stream tail after the failed burst");
        end
    end
    reg [31:0] hold_write_addr=0;
    integer overlapped_read_beats=0;
    always @(posedge clk) begin
        if(rst_n && hold_write_response && wr_addr_valid && wr_data_last && wr_addr==hold_write_addr)hold_hit<=1;
        if(rst_n && monitor_overlap && m_axi_arvalid && m_axi_arready && hold_write_response)overlapped_read_beats<=overlapped_read_beats+1;
    end
    integer bi,bj;
    always @(posedge clk) begin
        if (!rst_n) begin
            cycles<=0;random_state<=32'h4c734a91;
            rd_active<=0;m_axi_rvalid<=0;m_axi_bvalid<=0;
            wr_addr_valid<=0;wr_data_last<=0;wr_index<=0;b_delay<=0;
        end else begin
            cycles<=cycles+1;
            random_state<={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_araddr+((m_axi_arlen+1)*16)>MEM_BYTES ||
                    {1'b0,m_axi_araddr[11:0]}+(m_axi_arlen+1)*16>4096 ||
                    m_axi_arsize!=4 || m_axi_arburst!=1 || m_axi_arid!=0 || m_axi_araddr[3:0]!=0)
                    $fatal(1,"Illegal DDR read burst addr=%h len=%0d",m_axi_araddr,m_axi_arlen);
                rd_active<=1;rd_addr<=m_axi_araddr;rd_index<=0;rd_beats<=m_axi_arlen+1;
                read_bursts<=read_bursts+1;
            end
            if(rd_active&&held_read(rd_addr+rd_index*16))begin end
            else if(DENSE_DDR)begin
                if(m_axi_rvalid&&m_axi_rready)begin
                    rd_index<=rd_index+1;read_beats<=read_beats+1;
                    if(m_axi_rlast)begin m_axi_rvalid<=0;rd_active<=0;end
                    else begin
                        m_axi_rvalid<=!(held_read(rd_addr+(rd_index+1)*16));
                        m_axi_rlast<=(rd_index+1==rd_beats-1);
                        m_axi_rresp<=(inject_r_error&&!inject_r_done&&rd_addr+(rd_index+1)*16==inject_r_addr+16)?2'b10:2'b00;
                        for(bi=0;bi<16;bi=bi+1)m_axi_rdata[bi*8+:8]<=memory[rd_addr+(rd_index+1)*16+bi];
                    end
                end else if(!m_axi_rvalid&&rd_active)begin
                    m_axi_rvalid<=1;m_axi_rid<=0;m_axi_rlast<=(rd_index==rd_beats-1);
                    m_axi_rresp<=(inject_r_error&&!inject_r_done&&rd_addr+rd_index*16==inject_r_addr+16)?2'b10:2'b00;
                    for(bi=0;bi<16;bi=bi+1)m_axi_rdata[bi*8+:8]<=memory[rd_addr+rd_index*16+bi];
                end
            end else if (m_axi_rvalid) begin
                if (m_axi_rready) begin
                    m_axi_rvalid<=0;rd_index<=rd_index+1;read_beats<=read_beats+1;
                    if(m_axi_rlast) rd_active<=0;
                end
            end else if (rd_active && random_state[4:3]!=0) begin
                m_axi_rvalid<=1;m_axi_rresp<=(inject_r_error && !inject_r_done && rd_addr+rd_index*16==inject_r_addr+16)?2'b10:2'b00;m_axi_rid<=0;m_axi_rlast<=(rd_index==rd_beats-1);
                for (bi=0;bi<16;bi=bi+1) m_axi_rdata[bi*8+:8]<=memory[rd_addr+rd_index*16+bi];
            end
            if (m_axi_awvalid && m_axi_awready) begin
                if (m_axi_awaddr+((m_axi_awlen+1)*16)>MEM_BYTES ||
                    {1'b0,m_axi_awaddr[11:0]}+(m_axi_awlen+1)*16>4096 ||
                    m_axi_awsize!=4 || m_axi_awburst!=1 || m_axi_awid!=0 || m_axi_awaddr[3:0]!=0)
                    $fatal(1,"Illegal DDR write burst addr=%h len=%0d",m_axi_awaddr,m_axi_awlen);
                wr_addr_valid<=1;wr_addr<=m_axi_awaddr;wr_beats<=m_axi_awlen+1;
                write_bursts<=write_bursts+1;
            end
            if(m_axi_wvalid && m_axi_wready) begin
                if (m_axi_wstrb!=16'hffff || wr_index>=256) $fatal(1,"Illegal DDR write data");
                write_burst[wr_index]<=m_axi_wdata;
                wr_index<=wr_index+1;write_beats<=write_beats+1;
                if(m_axi_wlast) wr_data_last<=1;
            end
            if(wr_addr_valid && wr_data_last && !m_axi_bvalid) begin
                if(wr_index!=wr_beats) $fatal(1,"Incorrect DDR WLAST");
                if(b_delay<3) b_delay<=b_delay+1;
                else if(random_state[5] && !(hold_write_response && wr_addr==hold_write_addr) &&
                        (!write_error_target || !dut.u_core.u_writer.stream_empty)) begin
                    for(bi=0;bi<wr_beats;bi=bi+1)
                        for(bj=0;bj<16;bj=bj+1) memory[wr_addr+bi*16+bj]<=write_burst[bi][bj*8+:8];
                    m_axi_bvalid<=1;m_axi_bresp<=write_error_target?2'b10:2'b00;m_axi_bid<=0;
                end
            end
            if(m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid<=0;wr_addr_valid<=0;wr_data_last<=0;wr_index<=0;b_delay<=0;
            end
        end
    end
    always @(negedge clk) begin
        m_axi_arready = rst_n && !rd_active && !m_axi_rvalid && random_state[2:0]!=0;
        if(held_read(m_axi_araddr))
            m_axi_arready=0;
        m_axi_awready = rst_n && !wr_addr_valid && random_state[7:6]==0;
        m_axi_wready = rst_n && !hold_w_data && !wr_data_last && (DENSE_DDR||random_state[9:8]!=0);
    end
    reg stalled_aw=0,stalled_ar=0,stalled_w=0;
    reg [39:0] aw_hold,ar_hold;
    reg [144:0] w_hold;
    always @(posedge clk) begin
        if(rst_n) begin
            if(stalled_aw && (!m_axi_awvalid || {m_axi_awaddr,m_axi_awlen}!==aw_hold)) $fatal(1,"AW unstable under backpressure");
            if(stalled_ar && (!m_axi_arvalid || {m_axi_araddr,m_axi_arlen}!==ar_hold)) $fatal(1,"AR unstable under backpressure");
            if(stalled_w && (!m_axi_wvalid || {m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==w_hold)) $fatal(1,"W unstable under backpressure");
        end
        stalled_aw<=rst_n && m_axi_awvalid && !m_axi_awready;
        stalled_ar<=rst_n && m_axi_arvalid && !m_axi_arready;
        stalled_w<=rst_n && m_axi_wvalid && !m_axi_wready;
        aw_hold<={m_axi_awaddr,m_axi_awlen};ar_hold<={m_axi_araddr,m_axi_arlen};w_hold<={m_axi_wlast,m_axi_wstrb,m_axi_wdata};
    end

    task automatic cpu_write(input [7:0] addr,input [31:0] data,input [1:0] response);
        reg aw_ok,w_ok; integer timeout;
        begin
            @(negedge clk);S_AXI_AWADDR=AXIL_BASE+{24'd0,addr};
            S_AXI_WDATA=data;S_AXI_WSTRB=4'hf;
            S_AXI_AWVALID=1;S_AXI_WVALID=1;S_AXI_BREADY=1;
            aw_ok=0;w_ok=0;timeout=0;
            while(!aw_ok || !w_ok) begin
                @(posedge clk);timeout=timeout+1;
                if(S_AXI_AWVALID && S_AXI_AWREADY) aw_ok=1;
                if(S_AXI_WVALID && S_AXI_WREADY) w_ok=1;
                @(negedge clk);
                if(aw_ok) S_AXI_AWVALID=0;
                if(w_ok) S_AXI_WVALID=0;
                if(timeout>1000) $fatal(1,"AXIL write handshake timeout %h",addr);
            end
            timeout=0;
            while(!S_AXI_BVALID) begin @(negedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL B timeout %h",addr);end
            if(S_AXI_BRESP!==response) $fatal(1,"AXIL B response at %h expected %h got %h",addr,response,S_AXI_BRESP);
            @(posedge clk);@(negedge clk);S_AXI_BREADY=0;
        end
    endtask
    task automatic cpu_read(input [7:0] addr,output [31:0] data);
        integer timeout;
        begin
            @(negedge clk);S_AXI_ARADDR=AXIL_BASE+{24'd0,addr};S_AXI_ARVALID=1;S_AXI_RREADY=1;
            timeout=0;
            @(posedge clk);
            while(!S_AXI_ARREADY) begin @(posedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL AR timeout");end
            @(negedge clk);S_AXI_ARVALID=0;timeout=0;
            while(!S_AXI_RVALID) begin @(negedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL R timeout");end
            if(S_AXI_RRESP!==0) $fatal(1,"AXIL read error %h",addr);
            data=S_AXI_RDATA;
            @(posedge clk);@(negedge clk);S_AXI_RREADY=0;
        end
    endtask

    integer test_m,test_n,test_k,test_y_base=Y_BASE;
    integer saved_m[0:1],saved_n[0:1],saved_base[0:1];
    localparam integer EXPECTED_STRIDE=524288;
    integer expected[0:2*EXPECTED_STRIDE-1];
    integer a_read_bytes=0,bt_read_bytes=0;
    always @(posedge clk)if(rst_n&&m_axi_arvalid&&m_axi_arready)begin
        if(m_axi_araddr>=A_BASE&&m_axi_araddr<A_BASE+test_m*test_k)
            a_read_bytes<=a_read_bytes+(m_axi_arlen+1)*16;
        else if(m_axi_araddr>=BT_BASE&&m_axi_araddr<BT_BASE+test_n*test_k)
            bt_read_bytes<=bt_read_bytes+(m_axi_arlen+1)*16;
        else $fatal(1,"Read outside A/BT: %h",m_axi_araddr);
    end
    task automatic fill(input integer m,n,k,seed,extreme);
        integer r,c,kk,v;
        begin
            test_m=m;test_n=n;test_k=k;
            for(r=0;r<m;r=r+1)for(kk=0;kk<k;kk=kk+1)begin
                v=extreme?((r%2)?127:-128):((r*7+kk*5+(kk/3)*3+seed)%31)-15;
                memory[A_BASE+r*k+kk]=v;
            end
            for(c=0;c<n;c=c+1)for(kk=0;kk<k;kk=kk+1)begin
                v=extreme?((c%2)?127:-128):((c*11+kk*3+(kk/5)*7+seed*3)%29)-14;
                memory[BT_BASE+c*k+kk]=v;
            end
            for(r=test_y_base-32;r<test_y_base+m*n*4+64;r=r+1)memory[r]=8'ha5;
        end
    endtask
    task automatic save_reference(input integer slot);
        integer r,c,kk,a,b,sum;
        begin
            saved_m[slot]=test_m;saved_n[slot]=test_n;saved_base[slot]=test_y_base;
            for(r=0;r<test_m;r=r+1)for(c=0;c<test_n;c=c+1)begin
                sum=0;
                for(kk=0;kk<test_k;kk=kk+1)begin
                    a=$signed(memory[A_BASE+r*test_k+kk]);b=$signed(memory[BT_BASE+c*test_k+kk]);sum=sum+a*b;
                end
                expected[slot*EXPECTED_STRIDE+r*test_n+c]=sum;
            end
        end
    endtask
    task automatic compare_saved(input integer slot);
        integer r,c,v,errors,addr;
        reg [31:0] raw;
        begin
            errors=0;
            for(r=0;r<saved_m[slot];r=r+1)for(c=0;c<saved_n[slot];c=c+1)begin
                addr=saved_base[slot]+4*(r*saved_n[slot]+c);
                raw={memory[addr+3],memory[addr+2],memory[addr+1],memory[addr]};v=$signed(raw);
                if(v!==expected[slot*EXPECTED_STRIDE+r*saved_n[slot]+c])begin
                    if(errors<6)$display("row=%0d col=%0d want=%0d got=%0d",r,c,expected[slot*EXPECTED_STRIDE+r*saved_n[slot]+c],v);
                    errors=errors+1;
                end
            end
            if(errors)$fatal(1,"INT32 comparison failed slot%0d errors=%0d",slot,errors);
            for(r=saved_base[slot]-32;r<saved_base[slot];r=r+1)if(memory[r]!==8'ha5)$fatal(1,"leading guard overwritten");
            for(r=saved_base[slot]+saved_m[slot]*saved_n[slot]*4;r<saved_base[slot]+saved_m[slot]*saved_n[slot]*4+64;r=r+1)
                if(memory[r]!==8'ha5)$fatal(1,"trailing guard overwritten");
        end
    endtask
    task automatic configure;
        begin
            cpu_write(8'h04,A_BASE,0);cpu_write(8'h08,BT_BASE,0);cpu_write(8'h0c,test_y_base,0);
            cpu_write(8'h10,(test_m/32)|((test_n/32)<<16),0);cpu_write(8'h14,test_k/32,0);
        end
    endtask
    task automatic wait_complete(input bit want_error);
        integer timeout;reg [31:0] flags,job,done_job;
        begin
            timeout=0;flags=0;
            while(!(flags[2]&&!flags[0]&&!flags[6])&&!flags[7])begin
                repeat(32)@(negedge clk);cpu_read(8'h18,flags);timeout=timeout+1;
                if(timeout>40000)$fatal(1,"Completion timeout state%0d writer%0d",dut.u_core.state,dut.u_core.u_writer.state);
            end
            if(flags[2]!==want_error||flags[0]||flags[6])$fatal(1,"Unexpected FLAGS %h expectedError=%b",flags,want_error);
            if(!want_error)begin
                cpu_read(8'h20,job);cpu_read(8'h28,done_job);
                if(job!==done_job||!flags[1]||flags[3]||flags[5])$fatal(1,"INT32 complete/job/reserved FLAGS %h",flags);
            end else begin
                if(flags[7])$fatal(1,"fault also advertised WRITE_DONE");
                if(dut.u_core.produced_words!==0||dut.u_core.c_write_ptr!==0||dut.u_core.store_fire!==0)
                    $fatal(1,"failed output was not invalidated/cleared after drain");
            end
        end
    endtask
    task automatic run_good(input integer m,n,k,seed,extreme);
        integer run_start_cycle,run_start_read,run_start_write;
        reg [31:0] run_start_store;
        begin
            run_start_cycle=cycles;run_start_read=read_beats;run_start_write=write_beats;
            run_start_store=dut.u_core.store_q;
            test_y_base=Y_BASE;fill(m,n,k,seed,extreme);save_reference(0);configure();
            cpu_write(8'h00,32'h103,0);wait_complete(0);compare_saved(0);
            $display("PASS INT32 M=%0d N=%0d K=%0d",m,n,k);
            $display("PERF job_cycles=%0d interval_cycles=%0d total_cycles=%0d compute_cycles=%0d load_cycles=%0d store_cycles=%0d read_bytes=%0d write_bytes=%0d",
                dut.u_core.cycles_snap,cycles-run_start_cycle,cycles,dut.u_core.compute_snap,dut.u_core.load_snap,dut.u_core.store_q-run_start_store,
                (read_beats-run_start_read)*16,(write_beats-run_start_write)*16);
        end
    endtask
    reg early512_seen=0;
    integer capture_count=0;
    reg capture_started=0;
    always @(posedge clk)if(rst_n&&!dut.u_core.abort_all)begin
        if(dut.u_core.state==dut.u_core.SA_CLEAR||dut.u_core.state==dut.u_core.IDLE)begin
            capture_count=0;capture_started=0;
        end
        if(dut.u_core.dma_cmd_valid&&dut.u_core.dma_cmd_ready)begin
            if(dut.u_core.dma_cmd_bytes!=32*test_k)
                $fatal(1,"input DMA command does not cover the complete 32-row group");
        end
        if(dut.u_core.store_fire)begin
            if(!capture_started&&(dut.u_core.state!=dut.u_core.DRAIN||dut.u_core.drain_count!=17))
                $fatal(1,"first C capture did not start on drain17");
            if(dut.u_core.sa_row!=capture_count%32||dut.u_core.c_chunk!=capture_count/32)
                $fatal(1,"C chunk-major capture order");
            capture_started=1;capture_count=capture_count+1;
        end else if(capture_started&&capture_count<64)$fatal(1,"bubble inside 64-word C capture");
        if(dut.u_core.state==dut.u_core.WAIT_TILE&&capture_count!=64)
            $fatal(1,"C capture count mismatch");
    end
    integer committed_bt_words[0:7],committed_a_words[0:7];
    reg [63:0] seen_pairs=0;
    integer a_parallel_cycles=0,early_a_feed_cycles=0;
    integer cg,ctx_index=0,parallel_bank_cycles=0,background_feed_cycles=0,reads_while_w_blocked=0;
    always @(posedge clk)if(rst_n)begin
        if(dut.u_core.state==dut.u_core.TILE_SETUP)begin ctx_index=0;seen_pairs=0;end
        if(dut.u_core.task_active&&!dut.u_core.abort_all)begin
            // Count actual physical bank writes, independently of packer/DMA
            // tokens. A ready group may not omit even its final SRAM word.
            for(cg=0;cg<8;cg=cg+1)begin
                if(dut.u_core.bt_group_ready[cg]&&committed_bt_words[cg]!=32*dut.u_core.k_words)
                    $fatal(1,"premature/stale BT ready group=%0d",cg);
                if(dut.u_core.a_group_ready[cg]&&committed_a_words[cg]!=32*dut.u_core.k_words)
                    $fatal(1,"premature/stale A ready group=%0d",cg);
            end
            if(dut.u_core.state==dut.u_core.CTX_SEND&&dut.u_core.ctx_ready)begin
                if(!dut.u_core.a_group_ready[dut.u_core.mi/32]||!dut.u_core.bt_group_ready[dut.u_core.nj/32])
                    $fatal(1,"compute context issued before complete A/BT group");
                if(seen_pairs[(dut.u_core.nj/32)*8+dut.u_core.mi/32])$fatal(1,"duplicate ready pair");
                seen_pairs[(dut.u_core.nj/32)*8+dut.u_core.mi/32]=1;
                ctx_index=ctx_index+1;
            end
            if(dut.u_core.state==dut.u_core.FEED&&!dut.u_core.a_valid)early_a_feed_cycles=early_a_feed_cycles+1;
            if(dut.u_core.state==dut.u_core.FEED&&!dut.u_core.bt_valid)background_feed_cycles=background_feed_cycles+1;
            if(m_axi_rvalid&&m_axi_rready&&hold_w_data)reads_while_w_blocked=reads_while_w_blocked+1;
        end
        // Derive group independently from the physical linear SRAM address.
        if(|dut.u_core.bt_we)
            committed_bt_words[dut.u_core.u_bt.waddr/(32*dut.u_core.k_words)]=committed_bt_words[dut.u_core.u_bt.waddr/(32*dut.u_core.k_words)]+1;
        if(dut.u_core.a_we)
            committed_a_words[dut.u_core.u_a.waddr/(32*dut.u_core.k_words)]=committed_a_words[dut.u_core.u_a.waddr/(32*dut.u_core.k_words)]+1;
        if(dut.u_core.state==dut.u_core.CHECK_A&&!dut.u_core.a_hit)
            for(cg=0;cg<8;cg=cg+1)committed_a_words[cg]=0;
        // These tokens are cleared by the DUT at this edge (NBA), so retire
        // the old strip's scoreboard only after checking its old ready bits.
        if(dut.u_core.state==dut.u_core.CHECK_A&&!dut.u_core.bt_hit)
            for(cg=0;cg<8;cg=cg+1)committed_bt_words[cg]=0;
    end
    integer input_step=0,input_requests=0,ir,ik,expected_a_addr,expected_bt_addr;
    always @(posedge clk)if(rst_n&&!dut.u_core.abort_all)begin
        if(dut.u_core.a_we&&dut.u_core.input_read_valid)begin
            if(dut.u_core.u_a.waddr==dut.u_core.u_a.raddr)$fatal(1,"A same-address read/write collision");
            a_parallel_cycles=a_parallel_cycles+1;
        end
        if(dut.u_core.bt_we&&dut.u_core.input_read_valid)begin
            if(dut.u_core.u_bt.waddr==dut.u_core.u_bt.raddr)$fatal(1,"BT same-address read/write collision");
            parallel_bank_cycles=parallel_bank_cycles+1;
        end
        if(dut.u_core.state==dut.u_core.WORD_REQ)begin input_step=0;input_requests=0;end
        if(dut.u_core.input_read_valid)begin
            expected_a_addr=(dut.u_core.mi+input_requests%32)*(test_k/32)+input_requests/32;
            expected_bt_addr=(dut.u_core.nj+input_requests%32)*(test_k/32)+input_requests/32;
            if(input_requests>=test_k||dut.u_core.read_a_addr!=expected_a_addr||dut.u_core.read_bt_addr!=expected_bt_addr)
                $fatal(1,"single-port round-robin request/address mismatch request=%0d",input_requests);
            input_requests=input_requests+1;
        end else if(input_requests>0&&input_requests<test_k&&
            (dut.u_core.state==dut.u_core.WORD_WAIT||dut.u_core.state==dut.u_core.WORD_CAPTURE||
             dut.u_core.state==dut.u_core.FEED||dut.u_core.state==dut.u_core.DRAIN))
            $fatal(1,"bubble inserted inside continuous input prefetch");
        if(dut.u_core.state==dut.u_core.FEED||dut.u_core.state==dut.u_core.DRAIN)begin
            for(ir=0;ir<32;ir=ir+1)begin
                ik=input_step-ir;
                if(dut.u_core.a_lane_valid[ir]!==((ik>=0)&&(ik<test_k))||
                   dut.u_core.bt_lane_valid[ir]!==((ik>=0)&&(ik<test_k)))
                    $fatal(1,"input lane skew/valid mismatch step=%0d lane=%0d",input_step,ir);
                if(ik>=0&&ik<test_k)begin
                    if(dut.u_core.a_vec[ir*8+:8]!==memory[A_BASE+(dut.u_core.m0+dut.u_core.mi+ir)*test_k+ik]||
                       dut.u_core.bt_vec[ir*8+:8]!==memory[BT_BASE+(dut.u_core.n0+dut.u_core.nj+ir)*test_k+ik])
                        $fatal(1,"input lane byte order/K mismatch step=%0d lane=%0d",input_step,ir);
                end
            end
            input_step=input_step+1;
        end
        if(dut.u_core.state==dut.u_core.WAIT_TILE&&input_requests!=test_k)
            $fatal(1,"microtile did not read exactly K 256-bit words per operand");
    end
    // Predictions must have reached all local tiles on the same cycle that
    // the FSM consumes a byte. A fault may discard one final local operation.
    always @(posedge clk)if(rst_n&&!dut.u_core.abort_all)begin
        if(dut.u_core.u_array.local_advance!=={16{dut.u_core.state==dut.u_core.FEED||dut.u_core.state==dut.u_core.DRAIN}} ||
            dut.u_core.u_array.local_valid!=={16{dut.u_core.state==dut.u_core.FEED}})
            $fatal(1,"registered PE control/data phase misalignment");
    end
    always @(posedge clk)if(rst_n&&dut.u_core.state==dut.u_core.DRAIN&&dut.u_core.wr_cmd_valid)
        early512_seen<=1;
    integer c_committed[0:63];
    integer cs;
    reg [63:0] c_owned=0;
    integer c_reserve_seq=0;
    integer compute_write_overlap=0;
    // 每个 C 槽记录独立宏块代数，验证下一宏块可以在旧结果写完前加载、计算。
    integer macro_generation=0,c_macro_generation[0:63];
    integer cross_macro_load=0,cross_macro_compute=0,early_c_write_beats=0;
    always @(posedge clk)begin
        if(!rst_n||(dut.u_core.state==dut.u_core.IDLE&&dut.u_core.start_pulse&&!dut.u_core.error_q))begin
            for(cs=0;cs<64;cs=cs+1)c_committed[cs]=0;
            c_owned=0;c_reserve_seq=0;macro_generation=0;
        end else begin
        if(dut.u_core.state==dut.u_core.TILE_SETUP)macro_generation=macro_generation+1;
        if(!dut.u_core.abort_all&&m_axi_wvalid&&m_axi_wready&&!dut.u_core.ab_load_complete)
            early_c_write_beats=early_c_write_beats+1;
        if(!dut.u_core.abort_all&&dut.u_core.c_pending!=0&&
           c_macro_generation[dut.u_core.c_head]<macro_generation)begin
            if(m_axi_arvalid&&m_axi_arready)cross_macro_load=cross_macro_load+1;
            if(dut.u_core.state==dut.u_core.FEED)cross_macro_compute=cross_macro_compute+1;
        end
        if(dut.u_core.c_retire)begin
            if(!c_owned[dut.u_core.c_head]||c_committed[dut.u_core.c_head]!=64)
                $fatal(1,"C slot released before complete capture/DDR drain");
            c_owned[dut.u_core.c_head]=0;
        end
        if(dut.u_core.c_enqueue)begin
            if(c_owned[dut.u_core.c_tail]||dut.u_core.c_tail!=(c_reserve_seq%64))
                $fatal(1,"C circular slot reserved before previous writer retired");
            c_owned[dut.u_core.c_tail]=1;c_committed[dut.u_core.c_tail]=0;
            c_macro_generation[dut.u_core.c_tail]=macro_generation;
            c_reserve_seq=c_reserve_seq+1;
        end
        if(dut.u_core.store_fire)begin
            cs=dut.u_core.c_write_ptr/64;
            if(!c_owned[cs]||c_committed[cs]!=dut.u_core.c_write_ptr%64)
                $fatal(1,"C region overwritten or captured out of order slot=%0d",cs);
            c_committed[cs]=c_committed[cs]+1;
        end
        if(dut.u_core.c_req_valid)begin
            cs=dut.u_core.c_req_addr/64;
            if(dut.u_core.store_fire&&dut.u_core.c_req_addr==dut.u_core.c_write_ptr)
                $fatal(1,"C simultaneous read/write to same address");
            if(!c_owned[cs]||dut.u_core.c_req_addr%64>=c_committed[cs])
                $fatal(1,"writer read uncommitted C word slot=%0d address=%0d",cs,dut.u_core.c_req_addr);
        end
        if(!dut.u_core.abort_all&&dut.u_core.state==dut.u_core.FEED&&dut.u_core.writer_engine_live&&
           dut.u_core.c_head!=dut.u_core.c_micro_base[11:6])compute_write_overlap=compute_write_overlap+1;
        end
    end
    always @(posedge clk)if(rst_n&&dut.u_core.state==dut.u_core.WAIT_TILE&&!dut.u_core.abort_all)begin
        #1;
        // Result-selection payload may retain old data. The accumulators must
        // clear and no output word may be committed after invalidation.
        if(dut.u_core.u_array.accum[0]!==0||dut.u_core.u_array.accum[1023]!==0||
            dut.u_core.produced_words!==0||dut.u_core.store_fire!==0)
            $fatal(1,"PE accumulators/output validity not cleared after microtile C capture");
    end
    task automatic serial_hold(input bit fail_write);
        integer timeout;reg [31:0] f,j,j_after;
        begin
            test_y_base=Y_BASE;fill(32,128,32,1,1);save_reference(0);configure();
            hold_write_addr=Y_BASE;hold_write_response=1;hold_hit=0;
            cpu_write(0,32'h103,0);cpu_read(8'h20,j);timeout=0;
            while(!hold_hit||dut.u_core.state!=dut.u_core.WAIT_WRITE)begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>10000)$fatal(1,"compute did not finish macroblock while first B held");
            end
            cpu_read(8'h18,f);
            if(!f[0]||!f[6]||f[1]||f[7])$fatal(1,"premature completion while B held %h",f);
            if(dut.u_core.u_array.accum[0]!==0||ctx_index!=4||dut.u_core.c_pending!=4||dut.u_core.c_producer_live)
                $fatal(1,"computation waited for B instead of retaining four C regions");
            monitor_overlap=1;overlapped_read_beats=0;
            cpu_write(0,32'h103,2);cpu_write(4,A_BASE,2);cpu_read(8'h20,j_after);
            // Microtiles in this macroblock may all finish. A new job still
            // cannot overwrite configuration/C while DDR writes are pending.
            if(j_after!==j||ctx_index!=4||dut.u_core.c_pending!=4)
                $fatal(1,"new job accepted or C ownership lost while B held");
            if(fail_write)begin inject_b_addr=hold_write_addr;inject_b_error=1;inject_b_done=0;end
            @(negedge clk);hold_write_response=0;monitor_overlap=0;
            wait_complete(fail_write);
            if(fail_write)begin
                if(!inject_b_done||dut.u_core.u_writer.busy||dut.u_core.a_valid||dut.u_core.bt_valid||dut.u_core.c_pending)
                    $fatal(1,"write error did not drain active context and invalidate queued C");
                cpu_write(0,32'h103,2);cpu_write(0,32'h203,0);inject_b_error=0;
            end else begin
                compare_saved(0);
                if(dut.u_core.u_array.accum[0]!==0||dut.u_core.produced_words!==0)
                    $fatal(1,"outputs not synchronously cleared after final writeback");
            end
            $display("PASS async512 held B: all four microtiles captured, no premature job completion/START, writeError=%b",fail_write);
        end
    endtask
    task automatic c_capacity(input bit fail_write);
        integer timeout,slot;reg [31:0] f;
        begin
            test_y_base=Y_BASE;fill(fail_write?256:288,256,32,62,0);
            if(!fail_write)save_reference(0);
            configure();hold_write_addr=Y_BASE;hold_write_response=1;hold_hit=0;
            cpu_write(0,32'h103,0);timeout=0;
            while(!hold_hit||!(fail_write?(dut.u_core.state==dut.u_core.WAIT_WRITE):
                    (dut.u_core.state==dut.u_core.CTX_SEND&&dut.u_core.m0==256)))begin
                @(negedge clk);timeout=timeout+1;if(timeout>40000)$fatal(1,"full C macroblock capture timeout");
            end
            if(c_reserve_seq!=64||dut.u_core.c_pending!=64||dut.u_core.c_head!=0||dut.u_core.c_tail!=0||
               dut.u_core.c_write_ptr!=0||dut.u_core.c_producer_live||dut.u_core.u_array.accum[0]!==0)
                $fatal(1,"C full-capacity ownership or pointer wrap mismatch");
            for(slot=0;slot<64;slot=slot+1)
                if(c_committed[slot]!=64)$fatal(1,"incomplete C slot=%0d",slot);
            repeat(100)begin
                @(negedge clk);
                if(dut.u_core.store_fire||dut.u_core.c_pending!=64||c_reserve_seq!=64||dut.u_core.c_enqueue)
                    $fatal(1,"full C overwritten while DDR B held");
            end
            cpu_read(8'h18,f);
            if(!f[0]||!f[6]||f[1]||f[7])$fatal(1,"full C advertised premature job completion");
            if(fail_write)begin inject_b_addr=Y_BASE;inject_b_error=1;inject_b_done=0;end
            hold_write_response=0;wait_complete(fail_write);
            if(dut.u_core.c_pending||dut.u_core.c_inflight||dut.u_core.c_producer_live)
                $fatal(1,"C ownership survived completion/error cleanup");
            if(fail_write)begin
                if(!inject_b_done)$fatal(1,"full queue write error not injected");
                inject_b_error=0;cpu_write(0,32'h203,0);run_good(64,64,96,63,0);
            end else begin
                compare_saved(0);
                if(c_reserve_seq!=72||dut.u_core.completed_tiles!=2)
                    $fatal(1,"C circular reuse/macroblock retirement count mismatch");
            end
            $display("PASS C capacity64: no overwrite under B stall, circular reuse across macroblocks, writeError=%0d",fail_write);
        end
    endtask
    task automatic reset_c_queue;
        integer timeout;
        begin
            test_y_base=Y_BASE;fill(64,96,32,64,0);configure();hold_w_data=1;cpu_write(0,32'h103,0);timeout=0;
            while(dut.u_core.state!=dut.u_core.WAIT_WRITE||!m_axi_wvalid)begin
                @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"queued C reset phase timeout");
            end
            if(dut.u_core.c_pending!=6||!dut.u_core.c_inflight)$fatal(1,"reset test did not queue C contexts");
            rst_n=0;#1;if(dut.u_core.c_pending!=6)$fatal(1,"C queue reset was asynchronous");
            @(posedge clk);#1;
            if(dut.u_core.c_pending||dut.u_core.c_inflight||dut.u_core.c_producer_live||dut.u_core.task_active)
                $fatal(1,"C queue synchronous reset did not invalidate ownership");
            repeat(3)@(negedge clk);hold_w_data=0;rst_n=1;
            run_good(64,64,96,65,0);
            $display("PASS queued C reset: metadata only, stale payload/contexts not reused");
        end
    endtask
    task automatic older_write_error;
        integer timeout;
        begin
            test_y_base=Y_BASE;fill(64,96,256,66,0);configure();
            hold_write_addr=Y_BASE;hold_write_response=1;hold_hit=0;cpu_write(0,32'h103,0);timeout=0;
            while(!hold_hit||dut.u_core.state!=dut.u_core.FEED||dut.u_core.mi!=32)begin
                @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"older writer/next compute overlap timeout");
            end
            if(dut.u_core.c_pending<2||dut.u_core.c_head==dut.u_core.c_micro_base[11:6])
                $fatal(1,"older writer was not independent of current producer");
            inject_b_addr=Y_BASE;inject_b_error=1;inject_b_done=0;hold_write_response=0;
            wait_complete(1);
            if(!inject_b_done||dut.u_core.c_pending||dut.u_core.writer_live||dut.u_core.u_array.accum[0]!==0)
                $fatal(1,"older write error did not stop producer and flush pending contexts");
            inject_b_error=0;cpu_write(0,32'h203,0);run_good(64,64,96,67,0);
            $display("PASS older writer B error during next microtile FEED; global failure and recovery");
        end
    endtask
    task automatic reset_background_bt;
        integer timeout;
        begin
            test_y_base=Y_BASE;fill(64,96,256,50,0);configure();cpu_write(0,32'h103,0);timeout=0;
            while(!(dut.u_core.state==dut.u_core.FEED&&!dut.u_core.bt_valid&&dut.u_core.load_state==dut.u_core.LD_WAIT))begin
                @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"concurrent reset phase timeout");
            end
            rst_n=0;#1;
            if(!dut.u_core.task_active||!dut.u_core.bt_group_ready[0])$fatal(1,"loader reset was asynchronous");
            @(posedge clk);#1;
            if(dut.u_core.task_active||dut.u_core.a_group_ready||dut.u_core.bt_group_ready||dut.u_core.load_state!=dut.u_core.LD_IDLE||dut.u_core.ab_wr_valid||
               dut.u_core.c_pending||dut.u_core.c_inflight||dut.u_core.c_producer_live)
                $fatal(1,"synchronous reset failed to invalidate background loader");
            repeat(3)@(negedge clk);rst_n=1;
            run_good(64,64,96,51,0);
            $display("PASS reset during concurrent BT load/compute; fresh job replaced stale payload");
        end
    endtask
    task automatic early_bt;
        integer timeout,before_parallel,before_background,before_blocked;
        begin
            test_y_base=Y_BASE;fill(64,96,256,41,0);save_reference(0);configure();
            hold_bt_from=31;before_parallel=parallel_bank_cycles;
            before_background=background_feed_cycles;before_blocked=reads_while_w_blocked;
            cpu_write(0,32'h103,0);timeout=0;
            while(!(rd_active&&rd_addr+rd_index*16==BT_BASE+31*test_k))begin
                @(negedge clk);timeout=timeout+1;if(timeout>40000)$fatal(1,"BT row31 hold timeout");
            end
            repeat(30)begin
                @(negedge clk);
                if(ctx_index!=0||dut.u_core.bt_group_ready[0])$fatal(1,"started before 32 full BT rows");
            end
            hold_bt_from=32;timeout=0;
            // Even the first address of row33 is blocked: both A groups must
            // finish with just the first 32 BT rows, then wait for group1.
            while(!(dut.u_core.state==dut.u_core.WAIT_BT&&ctx_index==2))begin
                @(negedge clk);timeout=timeout+1;if(timeout>20000)$fatal(1,"first BT group did not traverse A");
            end
            if(ctx_index!=2||dut.u_core.bt_valid||dut.u_core.bt_group_ready!=1)
                $fatal(1,"first group required row33 or failed to reuse all A groups");
            repeat(20)begin
                @(negedge clk);
                if(dut.u_core.state!=dut.u_core.WAIT_BT||dut.u_core.sa_advance||ctx_index!=2)
                    $fatal(1,"waiting BT group advanced/repeated computation");
            end
            hold_bt_from=-1;hold_w_data=1;timeout=0;
            while(!dut.u_core.bt_valid)begin
                @(negedge clk);timeout=timeout+1;if(timeout>20000)$fatal(1,"W backpressure stopped BT loading");
            end
            repeat(80)@(negedge clk);
            if(!dut.u_core.writer_live||!m_axi_wvalid)$fatal(1,"W stall was not exercised");
            hold_w_data=0;wait_complete(0);compare_saved(0);
            if(early_a_feed_cycles==0||a_parallel_cycles==0||parallel_bank_cycles==before_parallel||background_feed_cycles==before_background||reads_while_w_blocked==before_blocked)
                $fatal(1,"missing concurrent BT bank read/write, early FEED, or R while W stalled");
            $display("PASS early BT: exact32 rows, A-group reuse, delayed groups, parallel banks=%0d, earlyFEED=%0d, RwithWblocked=%0d",
                parallel_bank_cycles-before_parallel,background_feed_cycles-before_background,reads_while_w_blocked-before_blocked);
        end
    endtask
    task automatic background_bt_error(input integer phase);
        integer timeout,writes_before;
        begin
            bg_error_phase=phase;
            test_y_base=Y_BASE;fill(64,96,(phase==3)?32:256,42+phase,0);configure();
            inject_r_addr=(phase==0)?BT_BASE:BT_BASE+32*test_k;inject_r_error=1;inject_r_done=0;
            hold_bt_from=(phase>=1)?32:-1;hold_w_data=(phase==3);writes_before=write_beats;
            cpu_write(0,32'h103,0);
            if(phase==1)begin
                timeout=0;
                while(!(dut.u_core.state==dut.u_core.FEED&&dut.u_core.mi==32&&dut.u_core.k_byte==8))begin
                    @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"BT error during FEED phase timeout");
                end
                hold_bt_from=-1;
            end
            if(phase==2)begin
                timeout=0;
                while(!(dut.u_core.state==dut.u_core.WAIT_BT&&ctx_index==2))begin
                    @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"early Y before BT error timeout");
                end
                if(write_beats==writes_before)$fatal(1,"error test did not first write partial Y");
                hold_bt_from=-1;
            end
            if(phase==3)begin
                timeout=0;
                while(!m_axi_wvalid)begin
                    @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"held W before BT error timeout");
                end
                hold_bt_from=-1;timeout=0;
                while(!dut.u_core.error_q)begin
                    @(negedge clk);timeout=timeout+1;if(timeout>10000)$fatal(1,"BT error while W held timeout");
                end
                repeat(20)begin
                    @(negedge clk);
                    if(!dut.u_core.task_active||!dut.u_core.writer_live||dut.u_core.write_done_seen)
                        $fatal(1,"background error released job before accepted W/B drain");
                end
                hold_w_data=0;
            end
            wait_complete(1);
            if(phase==0&&ctx_index!=0)$fatal(1,"first BT group error started computation");
            if(!inject_r_done||dut.u_core.a_group_ready||dut.u_core.bt_group_ready||dut.u_core.a_valid||dut.u_core.bt_valid||
               !dut.u_core.input_pack_empty||!dut.u_core.ab_writes_empty||dut.u_core.writer_live||dut.u_core.load_state!=dut.u_core.LD_IDLE)
                $fatal(1,"background BT error failed global invalidation/drain");
            inject_r_error=0;inject_r_addr=A_BASE;bg_error_phase=-1;cpu_write(0,32'h203,0);
            run_good(64,64,96,45+phase,0);
            $display("PASS BT error phase=%0d (before-ready/feed/partial-Y/held-W), global failure/drain and recovery",phase);
        end
    endtask
    task automatic early_a_exact;
        integer timeout;begin
            test_y_base=Y_BASE;fill(64,96,256,76,0);save_reference(0);configure();
            hold_a_from=31;cpu_write(0,32'h103,0);timeout=0;
            while(!(rd_active&&rd_addr+rd_index*16==A_BASE+31*test_k))begin
                @(negedge clk);timeout=timeout+1;if(timeout>40000)$fatal(1,"A row31 hold timeout");
            end
            repeat(30)begin @(negedge clk);if(ctx_index!=0||dut.u_core.a_group_ready)$fatal(1,"A group published before full32 rows");end
            hold_a_from=32;timeout=0;
            while(!(dut.u_core.state==dut.u_core.WAIT_BT&&ctx_index==1&&dut.u_core.c_pending!=0))begin
                @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"first A/BT pair waited for A row33");
            end
            if(dut.u_core.a_valid||dut.u_core.a_group_ready!=1||dut.u_core.bt_group_ready!=1)
                $fatal(1,"first ready pair required a full A strip");
            repeat(20)begin @(negedge clk);if(ctx_index!=1)$fatal(1,"unready A group used or pair repeated");end
            hold_a_from=-1;wait_complete(0);compare_saved(0);
            $display("PASS exact A32+BT32 start: no A row33 dependency, later independent A read/write, full INT32 reference");
        end
    endtask
    task automatic background_a_error(input integer phase);
        integer timeout;begin
            test_y_base=Y_BASE;fill(64,96,(phase==2)?32:256,80+phase,0);configure();
            inject_r_addr=(phase==0)?A_BASE:A_BASE+32*test_k;inject_r_error=1;inject_r_done=0;
            hold_a_from=(phase==0)?-1:32;hold_w_data=(phase==2);cpu_write(0,32'h103,0);
            timeout=0;
            if(phase==1)begin
                while(!(dut.u_core.state==dut.u_core.FEED&&dut.u_core.k_byte==8))begin
                    @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"A error FEED phase timeout");
                end
                hold_a_from=-1;
            end
            if(phase==2)begin
                while(!m_axi_wvalid)begin @(negedge clk);timeout=timeout+1;if(timeout>30000)$fatal(1,"A error held W timeout");end
                hold_a_from=-1;timeout=0;
                while(!dut.u_core.error_q)begin @(negedge clk);timeout=timeout+1;if(timeout>10000)$fatal(1,"A error not reported");end
                repeat(20)begin @(negedge clk);if(!dut.u_core.task_active||!dut.u_core.writer_live)$fatal(1,"A error retired before W drain");end
                hold_w_data=0;
            end
            wait_complete(1);
            if(!inject_r_done||(phase==0&&ctx_index!=0)||dut.u_core.a_group_ready||dut.u_core.bt_group_ready||dut.u_core.a_valid||dut.u_core.bt_valid)
                $fatal(1,"A error failed global invalidation");
            inject_r_error=0;inject_r_addr=A_BASE;cpu_write(0,32'h203,0);run_good(64,64,96,85+phase,0);
            $display("PASS A error phase=%0d, global failure, SRAM metadata invalidation, W/B drain and recovery",phase);
        end
    endtask
    task automatic errors;
        reg [31:0] v,id;integer reads_before;
        begin
            test_y_base=Y_BASE;fill(32,32,64,0,0);configure();
            reads_before=read_bursts;cpu_write(8'h10,32'h00010000,0);cpu_write(0,32'h103,0);wait_complete(1);
            if(read_bursts!=reads_before)$fatal(1,"invalid shape issued DDR traffic");
            cpu_read(8'h20,id);cpu_write(0,32'h103,2);cpu_read(8'h20,v);if(v!=id)$fatal(1,"ERROR accepted START");
            cpu_write(0,32'h203,0);
            configure();inject_r_error=1;inject_r_done=0;r_error_had_half=0;
            cpu_write(0,32'h103,0);wait_complete(1);inject_r_error=0;
            if(!inject_r_done||!r_error_had_half||!dut.u_core.input_pack_empty||!dut.u_core.ab_writes_empty)
                $fatal(1,"read-error residue did not drain");
            cpu_write(0,32'h203,0);run_good(32,32,64,3,0);
            fill(32,128,32,0,0);configure();inject_b_error=1;inject_b_addr=Y_BASE;inject_b_done=0;b_error_had_tail=0;
            cpu_write(0,32'h103,0);wait_complete(1);inject_b_error=0;
            if(!inject_b_done||!b_error_had_tail||dut.u_core.u_writer.busy)$fatal(1,"write error tail / ownership not drained");
            cpu_write(0,32'h203,0);run_good(32,32,32,4,1);
            configure();reads_before=read_bursts;cpu_write(4,32'hffffffe0,0);cpu_write(0,32'h103,0);wait_complete(1);
            if(read_bursts!=reads_before)$fatal(1,"overflow span issued DDR");
            cpu_write(0,32'h203,0);
            $display("PASS unified ERROR, read/write residue, START guard, recovery and span protection");
        end
    endtask
    task automatic pipeline_fault(input integer phase);
        integer timeout;
        begin
            test_y_base=Y_BASE;fill(32,32,64,20+phase,0);configure();
            cpu_write(0,32'h103,0);timeout=0;
            while(!((phase==0&&dut.u_core.state==dut.u_core.FEED&&dut.u_core.k_byte==17)||
                    (phase==1&&dut.u_core.state==dut.u_core.WORD_WAIT)||
                    (phase==2&&dut.u_core.state==dut.u_core.DRAIN&&
                     dut.u_core.produced_words>=3&&dut.u_core.u_writer.row_started)))begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>20000)$fatal(1,"pipeline fault injection phase not reached");
            end
            force dut.u_core.writer_fault=1'b1;
            #1;
            if(dut.u_core.store_fire||dut.u_core.dma_cmd_valid||dut.u_core.wr_cmd_valid)
                $fatal(1,"local control pipeline delayed fault commit suppression");
            @(posedge clk);@(negedge clk);release dut.u_core.writer_fault;
            wait_complete(1);
            if(dut.u_core.a_valid||dut.u_core.bt_valid||dut.u_core.u_array.accum[0]!==0)
                $fatal(1,"failed local pipeline survived cleanup");
            cpu_write(0,32'h203,0);
            run_good(32,32,96,30+phase,0);
            $display("PASS local-control fault phase=%0d, accepted writes drained, next job correct",phase);
        end
    endtask
    // 独立枚举四条蛇形路径，不使用 DUT 的成本公式。
    reg route_active=0;
    integer route_m[0:127],route_n[0:127];
    integer route_count=0,route_seen=0;
    always @(posedge clk)if(rst_n&&route_active&&dut.u_core.state==dut.u_core.TILE_SETUP)begin
        if(route_seen>=route_count||dut.u_core.m0!=route_m[route_seen]||dut.u_core.n0!=route_n[route_seen])
            $fatal(1,"snake macro path index=%0d expected=(%0d,%0d) actual=(%0d,%0d)",
                route_seen,route_m[route_seen],route_n[route_seen],dut.u_core.m0,dut.u_core.n0);
        route_seen=route_seen+1;
    end
    task automatic snake_reference(input integer m,n,k,want_route);
        integer cap,tm,tn,cm,cn,candidate,outer,inner,o,i,j,mr,nr,prm,prn,am,bn;
        integer cost,best,best_cost,chosen_a,chosen_b,a_before,b_before,write_before,cycle_before;
        begin
            cap=(524288/(32*k))*32;if(cap>256)cap=256;
            tm=(m<cap)?m:cap;tn=(n<cap)?n:cap;cm=(m+tm-1)/tm;cn=(n+tn-1)/tn;
            best=0;best_cost=2147483647;chosen_a=0;chosen_b=0;
            for(candidate=0;candidate<4;candidate=candidate+1)begin
                outer=(candidate/2)?cn:cm;inner=(candidate/2)?cm:cn;
                prm=-1;prn=-1;am=0;bn=0;
                for(o=0;o<outer;o=o+1)for(i=0;i<inner;i=i+1)begin
                    j=((candidate%2)^(o%2))?(inner-1-i):i;
                    mr=(candidate/2)?j*tm:o*tm;nr=(candidate/2)?o*tn:j*tn;
                    if(mr!=prm)am=am+((m-mr<tm)?m-mr:tm)*k;
                    if(nr!=prn)bn=bn+((n-nr<tn)?n-nr:tn)*k;
                    prm=mr;prn=nr;
                end
                cost=am+bn;
                if(cost<best_cost)begin best=candidate;best_cost=cost;chosen_a=am;chosen_b=bn;end
            end
            if(best!=want_route)$fatal(1,"test route expectation mismatch shape%0d/%0d best=%0d want=%0d",m,n,best,want_route);
            outer=(best/2)?cn:cm;inner=(best/2)?cm:cn;route_count=0;route_seen=0;
            for(o=0;o<outer;o=o+1)for(i=0;i<inner;i=i+1)begin
                j=((best%2)^(o%2))?(inner-1-i):i;
                route_m[route_count]=(best/2)?j*tm:o*tm;
                route_n[route_count]=(best/2)?o*tn:j*tn;
                route_count=route_count+1;
            end
            a_before=a_read_bytes;b_before=bt_read_bytes;write_before=write_beats;cycle_before=cycles;
            route_active=1;run_good(m,n,k,90+best,0);route_active=0;
            if(route_seen!=route_count||dut.u_core.completed_tiles!=route_count)
                $fatal(1,"snake did not visit/retire every macro exactly once visited=%0d retired=%0d want=%0d",
                    route_seen,dut.u_core.completed_tiles,route_count);
            if(a_read_bytes-a_before!=chosen_a||bt_read_bytes-b_before!=chosen_b)
                $fatal(1,"snake cache traffic A=%0d/%0d BT=%0d/%0d",a_read_bytes-a_before,chosen_a,bt_read_bytes-b_before,chosen_b);
            if(write_beats-write_before!=m*n/4)$fatal(1,"snake output byte traffic mismatch");
            $display("PASS snake route=%0d macros=%0d Abytes=%0d BTbytes=%0d minInputBytes=%0d cycles=%0d read_bytes=%0d write_bytes=%0d",
                best,route_count,chosen_a,chosen_b,best_cost,cycles-cycle_before,
                a_read_bytes-a_before+bt_read_bytes-b_before,(write_beats-write_before)*16);
        end
    endtask
    // 故意扣住第一块 C 的 B 响应：若仍存在宏块清空屏障，第二块无法开始。
    task automatic cross_macro_early_write;
        integer timeout,early_before,load_before,compute_before;
        begin
            test_y_base=Y_BASE;fill(288,64,32,97,0);save_reference(0);configure();
            early_before=early_c_write_beats;load_before=cross_macro_load;compute_before=cross_macro_compute;
            hold_write_addr=Y_BASE;hold_write_response=1;hold_hit=0;
            cpu_write(0,32'h103,0);timeout=0;
            while(!hold_hit||dut.u_core.state!=dut.u_core.WAIT_WRITE)begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>20000)$fatal(1,"cross-macro compute waited for older C writeback");
            end
            if(dut.u_core.c_pending!=18||c_reserve_seq!=18||dut.u_core.c_producer_live)
                $fatal(1,"cross-macro test did not retain all 18 C tiles while first B held");
            if(early_c_write_beats==early_before||cross_macro_load==load_before||cross_macro_compute==compute_before)
                $fatal(1,"missing early C write, next-macro load, or next-macro computation overlap");
            $display("PASS overlap: C-before-AB-complete beats=%0d, next-macro AR=%0d FEED=%0d while old C pending",
                early_c_write_beats-early_before,cross_macro_load-load_before,cross_macro_compute-compute_before);
            hold_write_response=0;wait_complete(0);compare_saved(0);
        end
    endtask
    integer before_a,before_b;reg [31:0] flags,v;
    initial begin
        repeat(5)@(negedge clk);rst_n=1;cpu_read(8'h50,v);if(v!==32'h0a080020)$fatal(1,"ABI10 CAP mismatch %h",v);
        case(CASE)
            0:begin run_good(32,32,32,0,1);run_good(32,96,96,1,0);run_good(64,64,64,2,0);run_good(32,32,32,7,0);
                if(!early512_seen)$fatal(1,"first 512-bit word did not start writeback during drain");end
            1:begin
                before_a=a_read_bytes;before_b=bt_read_bytes;run_good(32,288,32,1,0);
                if(a_read_bytes-before_a!=32*32||bt_read_bytes-before_b!=288*32)$fatal(1,"A residency traffic mismatch A=%0d BT=%0d",a_read_bytes-before_a,bt_read_bytes-before_b);
                cpu_read(8'h18,flags);if(flags[11:8]!=1||flags[15:12]!=8||flags[25:16]!=1)$fatal(1,"wrong auto dimensions %h",flags);
                $display("PASS A residency across N tail, no duplicate DMA");
            end
            2:begin
                run_good(32,32,16352,0,0);
                before_a=read_bursts;
                cpu_write(8'h14,512,0);cpu_write(0,32'h103,0);wait_complete(1);
                if(read_bursts!=before_a)$fatal(1,"K=16384 issued DDR");
                cpu_write(0,32'h203,0);
                cpu_write(8'h14,513,0);cpu_write(0,32'h103,0);wait_complete(1);
                if(read_bursts!=before_a)$fatal(1,"K=16416 issued DDR");
                cpu_write(0,32'h203,0);
                $display("PASS strict K<16384: maximum16352 correct, 16384/16416 rejected without DDR");
            end
            3:serial_hold(0);
            4:errors();
            5:begin
                before_a=a_read_bytes;before_b=bt_read_bytes;run_good(288,96,32,5,0);
                if(a_read_bytes-before_a!=288*32||bt_read_bytes-before_b!=96*32)$fatal(1,"BT reuse across M tail failed");
            end
            6:run_good(32,32,4096,2,0);
            7:begin serial_hold(1);run_good(32,32,32,8,0);end
            8:begin
                // Non-power-of-two output blocks and microtile row/column address transitions.
                run_good(96,96,96,9,0);run_good(160,32,32,10,0);
                run_good(192,32,32,11,0);run_good(224,32,32,12,0);
            end
            9:run_good(288,288,32,13,0); // Both M and N advance, including tails.
            10:begin pipeline_fault(0);pipeline_fault(1);pipeline_fault(2);end
            11:early_bt();
            12:begin background_bt_error(0);background_bt_error(1);background_bt_error(2);background_bt_error(3);end
            13:run_good(32,64,8192,48,0);
            14:run_good(64,96,1024,49,0);
            15:reset_background_bt();
            16:c_capacity(0);
            17:c_capacity(1);
            18:reset_c_queue();
            19:older_write_error();
            21:early_a_exact();
            22:begin background_a_error(0);background_a_error(1);background_a_error(2);end
            20:run_good(256,32,2048,71,0); // Fill A's full 512 KiB, including address16383.
            23:snake_reference(544,288,32,0); // 行蛇形，先正向，含双轴尾块。
            24:snake_reference(288,288,32,1); // 行蛇形，从 N 尾块开始，下一行复用完整 B。
            25:snake_reference(288,768,32,2); // 列蛇形，先正向；独立枚举证明读取量更少。
            26:snake_reference(288,512,32,3); // 列蛇形，从 M 尾块开始，下一列复用完整 A。
            27:cross_macro_early_write();
            default:$fatal(1,"unknown CASE");
        endcase
        if(rd_active&&held_read(rd_addr+rd_index*16))begin end
            else if(DENSE_DDR)begin
            if(max_r_streak<((test_k/16<DMA_BURST)?test_k/16:DMA_BURST))
                $fatal(1,"dense DDR did not exercise consecutive full-burst R beats");
            $display("PASS dense DDR: max R streak=%0d, no input FIFO, no master backpressure",max_r_streak);
        end
        $display("PASS tb_gemm_accelerator ABI10 CASE=%0d BURST=%0d cycles=%0d Rbeats=%0d Wbeats=%0d",CASE,DMA_BURST,cycles,read_beats,write_beats);$finish;
    end
    initial begin #60000000;$fatal(1,"Global timeout");end
endmodule
