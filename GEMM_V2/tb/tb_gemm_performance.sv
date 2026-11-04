`timescale 1ns/1ps
// 原 GEMM / GEMM_V2 共用的完整 RTL 性能平台；保留实际 1024 个 PE。
// A/BT 的内容与回归 fill(seed=61) 一致，CPU INT32 参考由 Python 预先生成。
// +REFERENCE=<hex> 指定按行排列、每行一个 32 位补码值的 C 参考文件。
module tb_gemm_performance #(
    parameter integer PM=288, PN=288, PK=2048,
    parameter integer DMA_BURST=256, DENSE_DDR=1,
    parameter integer MAX_CYCLES=20000000
);
    localparam [31:0] AXIL_BASE=32'h80000000;
    localparam [31:0] A_BASE=32'h00100000, BT_BASE=32'h01000000, Y_BASE=32'h02000000;
    localparam integer A_BYTES=PM*PK, BT_BYTES=PN*PK, C_BYTES=PM*PN*4;
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
    gemm_accelerator_top #(.MAX_BURST(DMA_BURST),.C_USE_URAM(1)) dut(.aclk(clk),.aresetn(rst_n),.*);

    // 三个分离数组表示稀疏 DDR 地址空间，避免为地址空洞分配几十 MiB。
    reg [7:0] a_memory[0:A_BYTES-1],bt_memory[0:BT_BYTES-1];
    reg [7:0] c_memory[0:C_BYTES+95]; // 前 32、后 64 字节为越界保护。
    reg [31:0] expected[0:PM*PN-1];
    reg [127:0] write_burst[0:255];
    string reference_file;
    reg [31:0] random_state=32'h4c734a91;
    integer cycles=0,read_bursts=0,write_bursts=0,read_beats=0,write_beats=0;
    integer a_read_bytes=0,bt_read_bytes=0;
    integer rd_index=0,rd_beats=0,wr_index=0,wr_beats=0,b_delay=0;
    reg rd_active=0,wr_addr_valid=0,wr_data_last=0;
    reg [31:0] rd_addr=0,wr_addr=0;
    integer r_streak=0,max_r_streak=0,max_pending=0,c_full_wait=0;
    integer bi,bj;
    function automatic bit valid_read(input [31:0] addr,input integer bytes);
        valid_read=((addr>=A_BASE)&&(addr+bytes<=A_BASE+A_BYTES))||
                   ((addr>=BT_BASE)&&(addr+bytes<=BT_BASE+BT_BYTES));
    endfunction
    function automatic [127:0] read_word(input [31:0] addr);
        integer q;
        begin
            read_word=0;
            for(q=0;q<16;q=q+1)begin
                if(addr>=A_BASE&&addr+16<=A_BASE+A_BYTES)
                    read_word[q*8+:8]=a_memory[addr-A_BASE+q];
                else if(addr>=BT_BASE&&addr+16<=BT_BASE+BT_BYTES)
                    read_word[q*8+:8]=bt_memory[addr-BT_BASE+q];
                else $fatal(1,"DDR read outside input addr=%h",addr);
            end
        end
    endfunction

    // 与现有集成回归相同的单读服务/单写服务模型，R/W 通道可重叠。
    // DENSE_DDR=1 连续给 R；=0 按相同 LFSR 插入空拍；B 响应仍有延迟。
    always @(posedge clk)begin
        if(!rst_n)begin
            cycles<=0;random_state<=32'h4c734a91;
            rd_active<=0;m_axi_rvalid<=0;m_axi_bvalid<=0;
            wr_addr_valid<=0;wr_data_last<=0;wr_index<=0;b_delay<=0;
        end else begin
            cycles<=cycles+1;
            random_state<={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
            if(cycles>MAX_CYCLES)$fatal(1,"performance timeout state=%0d",dut.u_core.state);
            if(m_axi_arvalid&&m_axi_arready)begin
                if(!valid_read(m_axi_araddr,(m_axi_arlen+1)*16)||
                   {1'b0,m_axi_araddr[11:0]}+(m_axi_arlen+1)*16>4096||
                   m_axi_arsize!=4||m_axi_arburst!=1||m_axi_arid!=0||m_axi_araddr[3:0]!=0)
                    $fatal(1,"illegal AR addr=%h len=%0d",m_axi_araddr,m_axi_arlen);
                rd_active<=1;rd_addr<=m_axi_araddr;rd_index<=0;rd_beats<=m_axi_arlen+1;
                read_bursts<=read_bursts+1;
                if(m_axi_araddr>=BT_BASE)bt_read_bytes<=bt_read_bytes+(m_axi_arlen+1)*16;
                else a_read_bytes<=a_read_bytes+(m_axi_arlen+1)*16;
            end
            if(DENSE_DDR)begin
                if(m_axi_rvalid&&m_axi_rready)begin
                    rd_index<=rd_index+1;read_beats<=read_beats+1;
                    if(m_axi_rlast)begin m_axi_rvalid<=0;rd_active<=0;end
                    else begin
                        m_axi_rvalid<=1;m_axi_rlast<=(rd_index+1==rd_beats-1);
                        m_axi_rdata<=read_word(rd_addr+(rd_index+1)*16);
                    end
                end else if(!m_axi_rvalid&&rd_active)begin
                    m_axi_rvalid<=1;m_axi_rid<=0;m_axi_rlast<=(rd_index==rd_beats-1);
                    m_axi_rdata<=read_word(rd_addr+rd_index*16);
                end
            end else if(m_axi_rvalid)begin
                if(m_axi_rready)begin
                    m_axi_rvalid<=0;rd_index<=rd_index+1;read_beats<=read_beats+1;
                    if(m_axi_rlast)rd_active<=0;
                end
            end else if(rd_active&&random_state[4:3]!=0)begin
                m_axi_rvalid<=1;m_axi_rid<=0;m_axi_rlast<=(rd_index==rd_beats-1);
                m_axi_rdata<=read_word(rd_addr+rd_index*16);
            end
            if(m_axi_awvalid&&m_axi_awready)begin
                if(m_axi_awaddr<Y_BASE||m_axi_awaddr+(m_axi_awlen+1)*16>Y_BASE+C_BYTES||
                   {1'b0,m_axi_awaddr[11:0]}+(m_axi_awlen+1)*16>4096||
                   m_axi_awsize!=4||m_axi_awburst!=1||m_axi_awid!=0||m_axi_awaddr[3:0]!=0)
                    $fatal(1,"illegal AW addr=%h len=%0d",m_axi_awaddr,m_axi_awlen);
                wr_addr_valid<=1;wr_addr<=m_axi_awaddr;wr_beats<=m_axi_awlen+1;
                write_bursts<=write_bursts+1;
            end
            // W 允许先于 AW 发送；收齐两者后才提交内存和 B 响应。
            if(m_axi_wvalid&&m_axi_wready)begin
                if(m_axi_wstrb!=16'hffff||wr_index>=256)$fatal(1,"illegal W data");
                write_burst[wr_index]<=m_axi_wdata;wr_index<=wr_index+1;write_beats<=write_beats+1;
                if(m_axi_wlast)wr_data_last<=1;
            end
            if(wr_addr_valid&&wr_data_last&&!m_axi_bvalid)begin
                if(wr_index!=wr_beats)$fatal(1,"W count/WLAST mismatch");
                if(b_delay<3)b_delay<=b_delay+1;
                else if(random_state[5])begin
                    for(bi=0;bi<wr_beats;bi=bi+1)
                        for(bj=0;bj<16;bj=bj+1)c_memory[32+wr_addr-Y_BASE+bi*16+bj]<=write_burst[bi][bj*8+:8];
                    m_axi_bvalid<=1;m_axi_bresp<=0;m_axi_bid<=0;
                end
            end
            if(m_axi_bvalid&&m_axi_bready)begin
                m_axi_bvalid<=0;wr_addr_valid<=0;wr_data_last<=0;wr_index<=0;b_delay<=0;
            end
            if(m_axi_rvalid&&!m_axi_rready)$fatal(1,"unexpected GEMM R backpressure");
            if(m_axi_rvalid&&m_axi_rready)begin
                r_streak=r_streak+1;if(r_streak>max_r_streak)max_r_streak=r_streak;
            end else r_streak=0;
            if(dut.u_core.c_pending>max_pending)max_pending=dut.u_core.c_pending;
            if(dut.u_core.state==dut.u_core.CTX_SEND&&!dut.u_core.ctx_ready)c_full_wait=c_full_wait+1;
        end
    end
    always @(negedge clk)begin
        m_axi_arready=rst_n&&!rd_active&&!m_axi_rvalid&&random_state[2:0]!=0;
        m_axi_awready=rst_n&&!wr_addr_valid&&random_state[7:6]==0;
        m_axi_wready=rst_n&&!wr_data_last&&(DENSE_DDR||random_state[9:8]!=0);
    end
    reg stalled_aw=0,stalled_ar=0,stalled_w=0;
    reg [39:0] aw_hold,ar_hold;
    reg [144:0] w_hold;
    always @(posedge clk)begin
        if(rst_n)begin
            if(stalled_aw&&(!m_axi_awvalid||{m_axi_awaddr,m_axi_awlen}!==aw_hold))$fatal(1,"AW unstable");
            if(stalled_ar&&(!m_axi_arvalid||{m_axi_araddr,m_axi_arlen}!==ar_hold))$fatal(1,"AR unstable");
            if(stalled_w&&(!m_axi_wvalid||{m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==w_hold))$fatal(1,"W unstable");
        end
        stalled_aw<=rst_n&&m_axi_awvalid&&!m_axi_awready;
        stalled_ar<=rst_n&&m_axi_arvalid&&!m_axi_arready;
        stalled_w<=rst_n&&m_axi_wvalid&&!m_axi_wready;
        aw_hold<={m_axi_awaddr,m_axi_awlen};ar_hold<={m_axi_araddr,m_axi_arlen};
        w_hold<={m_axi_wlast,m_axi_wstrb,m_axi_wdata};
    end

    task automatic cpu_write(input [7:0] addr,input [31:0] data);
        reg aw_ok,w_ok;integer timeout;
        begin
            @(negedge clk);S_AXI_AWADDR=AXIL_BASE+{24'd0,addr};S_AXI_WDATA=data;S_AXI_WSTRB=15;
            S_AXI_AWVALID=1;S_AXI_WVALID=1;S_AXI_BREADY=1;aw_ok=0;w_ok=0;timeout=0;
            while(!aw_ok||!w_ok)begin
                @(posedge clk);timeout=timeout+1;
                if(S_AXI_AWVALID&&S_AXI_AWREADY)aw_ok=1;
                if(S_AXI_WVALID&&S_AXI_WREADY)w_ok=1;
                @(negedge clk);
                if(aw_ok)S_AXI_AWVALID=0;if(w_ok)S_AXI_WVALID=0;
                if(timeout>1000)$fatal(1,"AXIL write timeout");
            end
            timeout=0;
            while(!S_AXI_BVALID)begin @(negedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL B timeout");end
            if(S_AXI_BRESP!=0)$fatal(1,"AXIL write error");
            @(posedge clk);@(negedge clk);S_AXI_BREADY=0;
        end
    endtask
    task automatic cpu_read(input [7:0] addr,output [31:0] data);
        integer timeout;
        begin
            @(negedge clk);S_AXI_ARADDR=AXIL_BASE+{24'd0,addr};S_AXI_ARVALID=1;S_AXI_RREADY=1;timeout=0;
            @(posedge clk);
            while(!S_AXI_ARREADY)begin @(posedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL AR timeout");end
            @(negedge clk);S_AXI_ARVALID=0;timeout=0;
            while(!S_AXI_RVALID)begin @(negedge clk);timeout=timeout+1;if(timeout>1000)$fatal(1,"AXIL R timeout");end
            if(S_AXI_RRESP!=0)$fatal(1,"AXIL read error");
            data=S_AXI_RDATA;@(posedge clk);@(negedge clk);S_AXI_RREADY=0;
        end
    endtask

    integer r,c,k,v,errors,idx;
    reg [31:0] flags,job,done_job,raw;
    initial begin
        if(PM<32||PN<32||PK<32||PM%32||PN%32||PK%32||PK>16352||
           A_BASE+A_BYTES>BT_BASE||BT_BASE+BT_BYTES>Y_BASE)$fatal(1,"invalid benchmark dimensions");
        if(!$value$plusargs("REFERENCE=%s",reference_file))$fatal(1,"missing +REFERENCE=<C int32 hex>");
        $readmemh(reference_file,expected);
        for(r=0;r<PM;r=r+1)for(k=0;k<PK;k=k+1)
            a_memory[r*PK+k]=((r*7+k*5+(k/3)*3+61)%31)-15;
        for(c=0;c<PN;c=c+1)for(k=0;k<PK;k=k+1)
            bt_memory[c*PK+k]=((c*11+k*3+(k/5)*7+61*3)%29)-14;
        for(idx=0;idx<C_BYTES+96;idx=idx+1)c_memory[idx]=8'ha5;
        repeat(5)@(negedge clk);rst_n=1;
        cpu_read(8'h50,flags);if(flags!==32'h0a080020)$fatal(1,"ABI mismatch");
        cpu_write(8'h04,A_BASE);cpu_write(8'h08,BT_BASE);cpu_write(8'h0c,Y_BASE);
        cpu_write(8'h10,(PM/32)|((PN/32)<<16));cpu_write(8'h14,PK/32);cpu_write(0,32'h103);
        flags=0;
        while(!flags[7])begin
            repeat(32)@(negedge clk);cpu_read(8'h18,flags);
            if(flags[2]&&!flags[0]&&!flags[6])$fatal(1,"GEMM error flags=%h",flags);
        end
        if(flags[0]||flags[2]||flags[6]||!flags[1])$fatal(1,"incorrect completion flags=%h",flags);
        cpu_read(8'h20,job);cpu_read(8'h28,done_job);if(job!==done_job)$fatal(1,"job ID mismatch");
        errors=0;
        for(idx=0;idx<PM*PN;idx=idx+1)begin
            raw={c_memory[32+idx*4+3],c_memory[32+idx*4+2],c_memory[32+idx*4+1],c_memory[32+idx*4]};
            if(raw!==expected[idx])begin
                if(errors<6)$display("C mismatch index=%0d expected=%h actual=%h",idx,expected[idx],raw);
                errors=errors+1;
            end
        end
        if(errors)$fatal(1,"INT32 mismatch count=%0d",errors);
        for(idx=0;idx<32;idx=idx+1)if(c_memory[idx]!==8'ha5)$fatal(1,"leading guard overwritten");
        for(idx=C_BYTES+32;idx<C_BYTES+96;idx=idx+1)if(c_memory[idx]!==8'ha5)$fatal(1,"trailing guard overwritten");
        if(write_beats*16!=C_BYTES||dut.u_core.write_bytes_q!=C_BYTES||
           read_beats*16!=a_read_bytes+bt_read_bytes||dut.u_core.read_bytes_q!=read_beats*16)
            $fatal(1,"DDR byte count mismatch");
        if(rd_active||m_axi_rvalid||wr_addr_valid||m_axi_bvalid)$fatal(1,"completion before DDR drain");
        $display("PERF M=%0d N=%0d K=%0d job_cycles=%0d total_cycles=%0d compute_cycles=%0d load_cycles=%0d store_cycles=%0d read_bytes=%0d write_bytes=%0d a_read_bytes=%0d bt_read_bytes=%0d read_bursts=%0d write_bursts=%0d max_pending=%0d c_full_wait=%0d max_r_streak=%0d",
            PM,PN,PK,dut.u_core.cycles_snap,cycles,dut.u_core.compute_snap,dut.u_core.load_snap,dut.u_core.store_q,
            read_beats*16,write_beats*16,a_read_bytes,bt_read_bytes,read_bursts,write_bursts,max_pending,c_full_wait,max_r_streak);
        $display("PASS tb_gemm_performance full RTL and complete INT32 reference");$finish;
    end
    // 仿真进度不参与流控；方便长矩阵的外部运行日志跟踪。
    always @(posedge clk)if(rst_n&&dut.u_core.task_active&&dut.u_core.cycles_q[17:0]==0)
        $display("PROGRESS job_cycles=%0d state=%0d",dut.u_core.cycles_q,dut.u_core.state);
endmodule
