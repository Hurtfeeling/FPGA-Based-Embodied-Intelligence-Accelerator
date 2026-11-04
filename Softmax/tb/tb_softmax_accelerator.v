`timescale 1ns/1ps
// Public-bus end-to-end test. The DDR model independently queues AR, AW and W,
// accepts W before AW, returns same-ID responses, and delays the final B.
module tb_softmax_accelerator #(
    parameter integer LARGE=0,
    parameter integer DMA_BURST=17,
    parameter integer OVERLAP=1
);
    localparam [31:0] AXIL_BASE=32'h80010000;
    localparam [31:0] SRC_ADDR=32'h00000ff0, DST_ADDR=32'h01000ff0;
    localparam integer MAX_BEATS=LARGE ? 262144 : 4096;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] S_AXI_AWADDR = 0;
    reg [2:0] S_AXI_AWPROT = 0;
    reg  S_AXI_AWVALID = 0;
    wire  S_AXI_AWREADY;
    reg [31:0] S_AXI_WDATA = 0;
    reg [3:0] S_AXI_WSTRB = 0;
    reg  S_AXI_WVALID = 0;
    wire  S_AXI_WREADY;
    wire [1:0] S_AXI_BRESP;
    wire  S_AXI_BVALID;
    reg  S_AXI_BREADY = 0;
    reg [31:0] S_AXI_ARADDR = 0;
    reg [2:0] S_AXI_ARPROT = 0;
    reg  S_AXI_ARVALID = 0;
    wire  S_AXI_ARREADY;
    wire [31:0] S_AXI_RDATA;
    wire [1:0] S_AXI_RRESP;
    wire  S_AXI_RVALID;
    reg  S_AXI_RREADY = 0;
    wire [0 : 0] m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7 : 0] m_axi_awlen;
    wire [2 : 0] m_axi_awsize;
    wire [1 : 0] m_axi_awburst;
    wire  m_axi_awlock;
    wire [3 : 0] m_axi_awcache;
    wire [2 : 0] m_axi_awprot;
    wire [3 : 0] m_axi_awqos;
    wire  m_axi_awvalid;
    reg  m_axi_awready = 0;
    wire [128-1 : 0] m_axi_wdata;
    wire [128/8-1 : 0] m_axi_wstrb;
    wire  m_axi_wlast;
    wire  m_axi_wvalid;
    reg  m_axi_wready = 0;
    reg [0 : 0] m_axi_bid = 0;
    reg [1 : 0] m_axi_bresp = 0;
    reg  m_axi_bvalid = 0;
    wire  m_axi_bready;
    wire [0 : 0] m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7 : 0] m_axi_arlen;
    wire [2 : 0] m_axi_arsize;
    wire [1 : 0] m_axi_arburst;
    wire  m_axi_arlock;
    wire [3 : 0] m_axi_arcache;
    wire [2 : 0] m_axi_arprot;
    wire [3 : 0] m_axi_arqos;
    wire  m_axi_arvalid;
    reg  m_axi_arready = 0;
    reg [0 : 0] m_axi_rid = 0;
    reg [128-1 : 0] m_axi_rdata = 0;
    reg [1 : 0] m_axi_rresp = 0;
    reg  m_axi_rlast = 0;
    reg  m_axi_rvalid = 0;
    wire  m_axi_rready;
    softmax_accelerator_top #(.MAX_BURST(DMA_BURST),
        .OVERLAP_STORE_EXP(OVERLAP),
        .EXP_LUT_FILE("Softmax/rtl/exp_lut.mem")) dut(.aclk(clk),.aresetn(rst_n),
        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWPROT(S_AXI_AWPROT),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),
        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),
        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY),
        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARPROT(S_AXI_ARPROT),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),
        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awqos(m_axi_awqos),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arqos(m_axi_arqos),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );
    reg [127:0] input_mem[0:MAX_BEATS-1],output_mem[0:MAX_BEATS-1];
    reg [8191:0] input_file,output_file;
    integer rows=1,cols=4,row_words,total_words;
    integer fault_mode=0,expect_code=0,config_mode=0,dense=0,inject_at=0;
    integer reset_mid=0,max_cycles=100000,final_b_delay=200;
    integer cycles=0,launch_cycle=0;
    reg [31:0] rng=32'h76543210;
    integer read_beats=0,write_beats=0,read_requests=0,write_requests=0;
    integer ar_beats=0,aw_beats=0,b_responses=0;
    integer peak_rd=0,peak_aw=0,w_before_aw=0,rw_overlap=0;
    integer first_aw_seen=-1;
    reg r_injected=0,b_injected=0,r_error_seen=0,b_error_seen=0;
    reg hold_r=0,hold_b=0;
    // FIFO of read descriptors, serving one ordered burst at a time.
    reg [31:0] r_addr[0:3],a_addr[0:3];
    integer r_len[0:3],a_len[0:3];
    integer rh=0,rt=0,rc=0,ri=0,ah=0,at=0,ac=0,ai=0;
    // W may arrive before its address. Buffer up to 64 data beats independently.
    reg [127:0] w_payload[0:63];
    reg w_last[0:63];
    integer wh=0,wt=0,wc=0;
    // Ordered B responses, with independent delays after the final W.
    reg [1:0] b_resp[0:3];
    integer b_due[0:3],bh=0,bt=0,bc=0;
    integer address_index,byte_count,i,j,fd,discard;
    reg [31:0] flags;
    reg [31:0] current_write_address;

    always @(posedge clk) begin
        if(!rst_n) begin
            cycles=0;rng=32'h76543210;rh=0;rt=0;rc=0;ri=0;
            ah=0;at=0;ac=0;ai=0;wh=0;wt=0;wc=0;bh=0;bt=0;bc=0;
            hold_r=0;hold_b=0;read_beats=0;write_beats=0;
            read_requests=0;write_requests=0;ar_beats=0;aw_beats=0;b_responses=0;
            peak_rd=0;peak_aw=0;w_before_aw=0;rw_overlap=0;first_aw_seen=-1;
            r_injected=0;b_injected=0;r_error_seen=0;b_error_seen=0;
        end else begin
            cycles=cycles+1;
            if(cycles>max_cycles)$fatal(1,"Simulation cycle limit reached: R=%0d W=%0d B=%0d",read_beats,write_beats,b_responses);
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            hold_r=m_axi_rvalid&&!m_axi_rready;
            hold_b=m_axi_bvalid&&!m_axi_bready;
            if(m_axi_awvalid&&first_aw_seen<0)first_aw_seen=cycles;
            if(m_axi_rvalid&&m_axi_rready&&m_axi_wvalid&&m_axi_wready)rw_overlap=rw_overlap+1;

            if(m_axi_bvalid&&m_axi_bready) begin
                if(bc==0)$fatal(1,"Unexpected B response handshake");
                if(m_axi_bresp!=0)b_error_seen=1;
                bh=(bh+1)%4;bc=bc-1;b_responses=b_responses+1;
            end
            if(m_axi_arvalid&&m_axi_arready) begin
                byte_count=(m_axi_arlen+1)*16;
                if(rc>=4||m_axi_arid!=0||m_axi_araddr[3:0]!=0||
                   m_axi_arsize!=4||m_axi_arburst!=1||m_axi_arlen>=DMA_BURST||
                   ({1'b0,m_axi_araddr[11:0]}+byte_count)>4096)
                    $fatal(1,"Illegal AR burst addr=%h len=%0d",m_axi_araddr,m_axi_arlen);
                if(m_axi_araddr!==SRC_ADDR+ar_beats*16||ar_beats+m_axi_arlen+1>total_words)
                    $fatal(1,"Read outside/gap in matrix: %h expected=%h",m_axi_araddr,SRC_ADDR+ar_beats*16);
                r_addr[rt]=m_axi_araddr;r_len[rt]=m_axi_arlen+1;
                rt=(rt+1)%4;rc=rc+1;ar_beats=ar_beats+m_axi_arlen+1;
                read_requests=read_requests+1;if(rc>peak_rd)peak_rd=rc;
            end
            if(m_axi_rvalid&&m_axi_rready) begin
                if(rc==0||m_axi_rlast!==(ri==r_len[rh]-1))$fatal(1,"R descriptor mismatch");
                if(m_axi_rresp!=0)begin r_injected=1;r_error_seen=1;end
                read_beats=read_beats+1;ri=ri+1;
                if(m_axi_rlast)begin rh=(rh+1)%4;rc=rc-1;ri=0;end
            end
            if(m_axi_awvalid&&m_axi_awready) begin
                byte_count=(m_axi_awlen+1)*16;
                if(ac>=4||m_axi_awid!=0||m_axi_awaddr[3:0]!=0||
                   m_axi_awsize!=4||m_axi_awburst!=1||m_axi_awlen>=DMA_BURST||
                   ({1'b0,m_axi_awaddr[11:0]}+byte_count)>4096)
                    $fatal(1,"Illegal AW burst addr=%h len=%0d",m_axi_awaddr,m_axi_awlen);
                if(m_axi_awaddr!==DST_ADDR+aw_beats*16||aw_beats+m_axi_awlen+1>total_words)
                    $fatal(1,"Write outside/gap in matrix: %h expected=%h",m_axi_awaddr,DST_ADDR+aw_beats*16);
                a_addr[at]=m_axi_awaddr;a_len[at]=m_axi_awlen+1;
                at=(at+1)%4;ac=ac+1;aw_beats=aw_beats+m_axi_awlen+1;
                write_requests=write_requests+1;if(ac>peak_aw)peak_aw=ac;
            end
            if(m_axi_wvalid&&m_axi_wready) begin
                if(wc>=64||m_axi_wstrb!==16'hffff)$fatal(1,"W queue overflow or unexpected WSTRB");
                if(ac==0)w_before_aw=w_before_aw+1;
                w_payload[wt]=m_axi_wdata;w_last[wt]=m_axi_wlast;
                wt=(wt+1)%64;wc=wc+1;write_beats=write_beats+1;
            end
            if(ac>0&&wc>0) begin
                current_write_address=a_addr[ah]+ai*16;
                address_index=(current_write_address-DST_ADDR)/16;
                if(address_index<0||address_index>=total_words)$fatal(1,"Write range overflow");
                if(w_last[wh]!==(ai==a_len[ah]-1))$fatal(1,"Incorrect WLAST for AWLEN");
                output_mem[address_index]=w_payload[wh];
                if(w_last[wh])begin
                    if(bc>=4)$fatal(1,"B response queue overflow");
                    b_resp[bt]=(fault_mode==2&&!b_injected)?2'b10:2'b00;
                    if(fault_mode==2&&!b_injected)b_injected=1;
                    b_due[bt]=cycles+3+(dense?0:rng[7:4]);
                    if(address_index==total_words-1)b_due[bt]=b_due[bt]+final_b_delay;
                    bt=(bt+1)%4;bc=bc+1;ah=(ah+1)%4;ac=ac-1;ai=0;
                end else ai=ai+1;
                wh=(wh+1)%64;wc=wc-1;
            end
        end
    end

    // Slave outputs are driven on the opposite edge, so neither bus endpoint
    // races the other at the sampling edge. Held R/B payloads stay unchanged.
    always @(negedge clk) begin
        if(!rst_n)begin
            m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;
            m_axi_rvalid=0;m_axi_bvalid=0;
            m_axi_rdata=0;m_axi_rresp=0;m_axi_rlast=0;m_axi_rid=0;
            m_axi_bresp=0;m_axi_bid=0;
        end else begin
            m_axi_arready=(rc<4)&&(dense||rng[2:0]!=0);
            m_axi_awready=(ac<4)&&(dense||rng[4:3]!=0)&&
                (write_requests!=0||(first_aw_seen>=0&&cycles-first_aw_seen>=12));
            m_axi_wready=(wc<64)&&(dense||rng[6:5]!=0);
            if(!hold_r)begin
                m_axi_rvalid=(rc>0)&&(dense||rng[9:7]!=0);
                if(m_axi_rvalid)begin
                    address_index=(r_addr[rh]-SRC_ADDR)/16+ri;
                    if(address_index<0||address_index>=total_words)$fatal(1,"Read memory index out of bounds");
                    m_axi_rdata=input_mem[address_index];m_axi_rid=0;
                    m_axi_rlast=(ri==r_len[rh]-1);
                    m_axi_rresp=(fault_mode==1&&!r_injected&&read_beats==inject_at)?2'b10:2'b00;
                end
            end
            if(!hold_b)begin
                m_axi_bvalid=(bc>0)&&(cycles>=b_due[bh])&&(dense||rng[12:10]!=0);
                if(m_axi_bvalid)begin m_axi_bresp=b_resp[bh];m_axi_bid=0;end
            end
        end
    end

    reg stalled_aw=0,stalled_ar=0,stalled_w=0;
    reg [39:0] aw_hold,ar_hold;
    reg [144:0] w_hold;
    always @(posedge clk)begin
        if(rst_n)begin
            if(stalled_aw&&(!m_axi_awvalid||{m_axi_awaddr,m_axi_awlen}!==aw_hold))
                $fatal(1,"AW changed under backpressure");
            if(stalled_ar&&(!m_axi_arvalid||{m_axi_araddr,m_axi_arlen}!==ar_hold))
                $fatal(1,"AR changed under backpressure");
            if(stalled_w&&(!m_axi_wvalid||{m_axi_wlast,m_axi_wstrb,m_axi_wdata}!==w_hold))
                $fatal(1,"W changed under backpressure");
        end
        stalled_aw<=rst_n&&m_axi_awvalid&&!m_axi_awready;
        stalled_ar<=rst_n&&m_axi_arvalid&&!m_axi_arready;
        stalled_w<=rst_n&&m_axi_wvalid&&!m_axi_wready;
        aw_hold<={m_axi_awaddr,m_axi_awlen};ar_hold<={m_axi_araddr,m_axi_arlen};
        w_hold<={m_axi_wlast,m_axi_wstrb,m_axi_wdata};
    end

    task automatic cpu_write(input [3:0] offset,input [31:0] data);
        reg aw_ok,w_ok;integer timeout;
        begin
            @(negedge clk);S_AXI_AWADDR=AXIL_BASE+offset;
            S_AXI_WDATA=data;S_AXI_WSTRB=15;S_AXI_AWVALID=1;S_AXI_WVALID=1;S_AXI_BREADY=1;
            aw_ok=0;w_ok=0;timeout=0;
            while(!aw_ok||!w_ok)begin
                @(posedge clk);timeout=timeout+1;
                if(S_AXI_AWVALID&&S_AXI_AWREADY)aw_ok=1;
                if(S_AXI_WVALID&&S_AXI_WREADY)w_ok=1;
                @(negedge clk);
                if(aw_ok)S_AXI_AWVALID=0;if(w_ok)S_AXI_WVALID=0;
                if(timeout>1000)$fatal(1,"AXIL write handshake timeout");
            end
            timeout=0;
            while(!S_AXI_BVALID)begin @(negedge clk);timeout=timeout+1;
                if(timeout>1000)$fatal(1,"AXIL write response timeout");end
            if(S_AXI_BRESP!==0)$fatal(1,"Unexpected AXIL write response offset=%h resp=%b",offset,S_AXI_BRESP);
            @(posedge clk);@(negedge clk);S_AXI_BREADY=0;
        end
    endtask
    task automatic cpu_read(output [31:0] data);
        integer timeout;
        begin
            @(negedge clk);S_AXI_ARADDR=AXIL_BASE;S_AXI_ARVALID=1;S_AXI_RREADY=1;
            @(posedge clk);timeout=0;
            while(!S_AXI_ARREADY)begin @(posedge clk);timeout=timeout+1;
                if(timeout>1000)$fatal(1,"AXIL read handshake timeout");end
            @(negedge clk);S_AXI_ARVALID=0;timeout=0;
            while(!S_AXI_RVALID)begin @(negedge clk);timeout=timeout+1;
                if(timeout>1000)$fatal(1,"AXIL read response timeout");end
            if(S_AXI_RRESP!==0)$fatal(1,"AXIL read response error");
            data=S_AXI_RDATA;
            @(posedge clk);@(negedge clk);S_AXI_RREADY=0;
        end
    endtask
    task automatic configure(input integer mode);
        reg [31:0] source,destination,dimension;
        begin
            source=SRC_ADDR;destination=DST_ADDR;dimension=((cols-1)<<16)|(rows-1);
            case(mode)
                1:dimension=(32'h1000<<16)|(rows-1); // K=4097
                2:source=SRC_ADDR+1;
                3:destination=SRC_ADDR; // overlap
                4:source=32'hfffffff0; // end-address overflow
                default:begin end
            endcase
            cpu_write(4,source);cpu_write(8,destination);cpu_write(12,dimension);
            launch_cycle=cycles;cpu_write(0,32'h101);
        end
    endtask
    task automatic wait_complete(input integer wanted_error);
        integer polls;
        reg [31:0] status;
        begin
            polls=0;status=0;
            while((!status[2]&&!status[3])||status[1])begin
                repeat(16)@(negedge clk);cpu_read(status);polls=polls+1;
                if(status[2]&&status[1])$fatal(1,"DONE asserted before BUSY cleared");
                if(polls>max_cycles/16)$fatal(1,"AXIL completion timeout");
            end
            if(status[1])$fatal(1,"Completion while BUSY: %h",status);
            if(wanted_error==0)begin
                if(!status[2]||status[3])$fatal(1,"Unexpected job error: %h",status);
                if(read_beats!=total_words||write_beats!=total_words||
                   ar_beats!=total_words||aw_beats!=total_words)
                    $fatal(1,"Incorrect matrix traffic R=%0d W=%0d expected=%0d",read_beats,write_beats,total_words);
            end else begin
                if(!status[3]||status[2]||status[7:4]!=wanted_error)
                    $fatal(1,"Error code got=%h expected=%0d",status,wanted_error);
                if(fault_mode==1&&!r_error_seen)$fatal(1,"Requested R error was not exercised");
                if(fault_mode==2&&!b_error_seen)$fatal(1,"Requested B error was not exercised");
                if(config_mode!=0&&(read_requests!=0||write_requests!=0))
                    $fatal(1,"Invalid configuration reached AXI");
            end
            if(rc||ac||wc||bc||m_axi_rvalid||m_axi_bvalid||b_responses!=write_requests)
                $fatal(1,"DONE/ERROR before AXI drain: Rq=%0d Aq=%0d Wq=%0d Bq=%0d B=%0d AW=%0d",
                    rc,ac,wc,bc,b_responses,write_requests);
        end
    endtask
    task automatic reset_statistics;
        begin
            @(negedge clk);
            if(rc||ac||wc||bc)$fatal(1,"Statistics reset with pending AXI transactions");
            read_beats=0;write_beats=0;read_requests=0;write_requests=0;
            ar_beats=0;aw_beats=0;b_responses=0;peak_rd=0;peak_aw=0;
            w_before_aw=0;rw_overlap=0;first_aw_seen=-1;
            r_injected=0;b_injected=0;r_error_seen=0;b_error_seen=0;
        end
    endtask

    initial begin
        if(!$value$plusargs("INPUT=%s",input_file))$fatal(1,"INPUT file is required");
        if(!$value$plusargs("OUTPUT=%s",output_file))$fatal(1,"OUTPUT file is required");
        discard=$value$plusargs("ROWS=%d",rows);
        discard=$value$plusargs("COLS=%d",cols);
        discard=$value$plusargs("FAULT=%d",fault_mode);
        discard=$value$plusargs("EXPECT_CODE=%d",expect_code);
        discard=$value$plusargs("CONFIG=%d",config_mode);
        discard=$value$plusargs("DENSE=%d",dense);
        discard=$value$plusargs("INJECT_AT=%d",inject_at);
        discard=$value$plusargs("RESET_MID=%d",reset_mid);
        discard=$value$plusargs("MAX_CYCLES=%d",max_cycles);
        discard=$value$plusargs("FINAL_B_DELAY=%d",final_b_delay);
        row_words=(cols+3)/4;total_words=rows*row_words;
        if(rows<1||rows>65536||cols<1||cols>4096||total_words>MAX_BEATS)
            $fatal(1,"Testbench memory/dimension limit");
        $readmemh(input_file,input_mem,0,total_words-1);
        for(i=0;i<total_words;i=i+1)output_mem[i]={4{32'hdeadbeef}};
        repeat(5)@(negedge clk);rst_n=1;
        configure(config_mode);
        if(reset_mid)begin
            // Abort an active matrix, reset both endpoints, then restart.
            repeat(35)@(negedge clk);
            rst_n=0;S_AXI_ARVALID=0;S_AXI_AWVALID=0;S_AXI_WVALID=0;
            S_AXI_BREADY=0;S_AXI_RREADY=0;
            repeat(5)@(negedge clk);rst_n=1;
            cpu_read(flags);if(flags!==0)$fatal(1,"Reset did not clear status");
            configure(0);
        end
        wait_complete(expect_code);
        if(expect_code!=0)begin
            $display("PASS expected error code=%0d, drained AXI; checking CLEAR and restart",expect_code);
            cpu_write(0,32'h201);cpu_read(flags);
            if(flags!==1)$fatal(1,"CLEAR did not clear sticky status: %h",flags);
            fault_mode=0;config_mode=0;
            // Recovery uses uniform rows, preserving NaN padding to check keep.
            for(i=0;i<total_words;i=i+1)begin
                output_mem[i]={4{32'hdeadbeef}};
                for(j=0;j<4;j=j+1)
                    input_mem[i][j*32+:32]=((i%row_words)*4+j<cols)?32'd0:32'h7fc00001;
            end
            reset_statistics();configure(0);wait_complete(0);
        end
        for(i=0;i<rows;i=i+1)for(j=cols;j<row_words*4;j=j+1)
            if(output_mem[i*row_words+j/4][(j%4)*32+:32]!==0)
                $fatal(1,"Output padding is not +0");
        fd=$fopen(output_file,"w");
        if(fd==0)$fatal(1,"Could not open output file");
        for(i=0;i<total_words;i=i+1)$fdisplay(fd,"%032h",output_mem[i]);
        $fclose(fd);
        $display("PASS AXI matrix M=%0d K=%0d cycles=%0d R=%0d W=%0d AR=%0d AW=%0d B=%0d maxRd=%0d maxAw=%0d WbeforeAW=%0d RWoverlap=%0d",
            rows,cols,cycles-launch_cycle,read_beats,write_beats,read_requests,write_requests,b_responses,peak_rd,peak_aw,w_before_aw,rw_overlap);
        $finish;
    end
endmodule
