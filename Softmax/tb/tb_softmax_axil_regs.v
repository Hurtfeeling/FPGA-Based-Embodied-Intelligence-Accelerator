`timescale 1ns/1ps
module tb_softmax_axil_regs;
    localparam [31:0] BASE=32'ha0010000;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,busy=0,done=0,error=0;
    reg [3:0] error_code=0;
    wire cfg_enable,start_pulse,clear_status_pulse;
    wire [31:0] cfg_src,cfg_dst,cfg_dim;
    reg [31:0] awaddr=0,wdata=0,araddr=0;
    reg [3:0] wstrb=0;
    reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
    wire awready,wready,bvalid,arready,rvalid;
    wire [1:0] bresp,rresp;
    wire [31:0] rdata;
    integer starts=0,clears=0,i;
    reg [96:0] held_cfg;
    softmax_axil_regs #(.C_S_AXI_BASEADDR(BASE)) dut(
        .busy(busy),.done(done),.error(error),.error_code(error_code),
        .cfg_enable(cfg_enable),.cfg_src(cfg_src),.cfg_dst(cfg_dst),.cfg_dim(cfg_dim),
        .start_pulse(start_pulse),.clear_status_pulse(clear_status_pulse),
        .S_AXI_ACLK(clk),.S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr),.S_AXI_AWPROT(3'd0),.S_AXI_AWVALID(awvalid),.S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),.S_AXI_WSTRB(wstrb),.S_AXI_WVALID(wvalid),.S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),.S_AXI_BVALID(bvalid),.S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),.S_AXI_ARPROT(3'd0),.S_AXI_ARVALID(arvalid),.S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),.S_AXI_RRESP(rresp),.S_AXI_RVALID(rvalid),.S_AXI_RREADY(rready));

    always @(posedge clk) begin
        if (!rst_n) begin starts=0;clears=0;end
        else begin
            if(start_pulse)starts=starts+1;
            if(clear_status_pulse)clears=clears+1;
        end
    end

    task automatic send_aw(input [31:0] address);
        begin
            @(negedge clk);awaddr=address;awvalid=1;
            @(posedge clk);while(!awready)@(posedge clk);
            @(negedge clk);awvalid=0;
        end
    endtask
    task automatic send_w(input [31:0] data,input [3:0] strobe);
        begin
            @(negedge clk);wdata=data;wstrb=strobe;wvalid=1;
            @(posedge clk);while(!wready)@(posedge clk);
            @(negedge clk);wvalid=0;
        end
    endtask
    task automatic begin_write(input [31:0] address,input [31:0] data,input [3:0] strobe,input integer order);
        begin
            if(order==1)begin send_aw(address);repeat(2)@(posedge clk);send_w(data,strobe);end
            else if(order==2)begin send_w(data,strobe);repeat(2)@(posedge clk);send_aw(address);end
            else begin fork send_aw(address);send_w(data,strobe);join end
        end
    endtask
    task automatic finish_write(input [1:0] expected_resp);
        reg [1:0] held;integer timeout;
        begin
            timeout=0;
            while(!bvalid&&timeout<40)begin @(posedge clk);#1;timeout=timeout+1;end
            if(!bvalid||bresp!==expected_resp)
                $fatal(1,"AXIL write addr=%h got=%b expected=%b",awaddr,bresp,expected_resp);
            held=bresp;
            repeat(3)begin @(posedge clk);#1;
                if(!bvalid||bresp!==held||awready||wready)
                    $fatal(1,"AXIL B response changed under backpressure");
            end
            @(negedge clk);bready=1;
            @(posedge clk);@(negedge clk);bready=0;
        end
    endtask
    task automatic write_abs(input [31:0] address,input [31:0] data,input [3:0] strobe,input integer order,input [1:0] response);
        begin begin_write(address,data,strobe,order);finish_write(response);end
    endtask
    task automatic wr(input [31:0] offset,input [31:0] data,input [3:0] strobe,input integer order,input [1:0] response);
        begin write_abs(BASE+offset,data,strobe,order,response);end
    endtask
    task automatic rd_abs(input [31:0] address,input [31:0] expected,input [1:0] response);
        reg [31:0] held;reg [1:0] held_resp;integer timeout;
        begin
            @(negedge clk);araddr=address;arvalid=1;
            @(posedge clk);while(!arready)@(posedge clk);
            @(negedge clk);arvalid=0;timeout=0;
            while(!rvalid&&timeout<20)begin @(posedge clk);#1;timeout=timeout+1;end
            if(!rvalid||rdata!==expected||rresp!==response)
                $fatal(1,"AXIL read addr=%h got=%h/%b expected=%h/%b",address,rdata,rresp,expected,response);
            held=rdata;held_resp=rresp;
            repeat(3)begin @(posedge clk);#1;
                if(!rvalid||rdata!==held||rresp!==held_resp||arready)
                    $fatal(1,"AXIL R response changed under backpressure");
            end
            @(negedge clk);rready=1;
            @(posedge clk);@(negedge clk);rready=0;
        end
    endtask
    task automatic rd(input [31:0] offset,input [31:0] expected);
        begin rd_abs(BASE+offset,expected,0);end
    endtask

    initial begin
        repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
        for(i=0;i<4;i=i+1)rd(i*4,0);
        // AW first, W first, and simultaneous acceptance.
        wr(4,32'h11223344,15,1,0);rd(4,32'h11223344);
        wr(8,32'h55667788,15,2,0);rd(8,32'h55667788);
        wr(12,32'h03ff03ff,15,0,0);rd(12,32'h03ff03ff);
        wr(4,32'haa,1,1,0);wr(4,32'hbb00,2,2,0);
        wr(4,32'hcc0000,4,0,0);wr(4,32'hdd000000,8,1,0);
        rd(4,32'hddccbbaa);
        held_cfg={cfg_enable,cfg_src,cfg_dst,cfg_dim};
        wr(4,32'hffffffff,0,2,0);
        if({cfg_enable,cfg_src,cfg_dst,cfg_dim}!==held_cfg)$fatal(1,"zero WSTRB changed configuration");
        write_abs(BASE-4,0,15,0,2);wr(16,0,15,1,2);wr(2,0,15,2,2);
        write_abs(BASE+32'h10000004,0,15,0,2);
        write_abs(32'hffffffff,32'hffffffff,0,2,2);
        rd_abs(BASE-4,0,2);rd_abs(BASE+16,0,2);rd_abs(BASE+2,0,2);
        rd_abs(BASE+32'h10000004,0,2);
        // START requires ENABLE. A byte-one write preserves ENABLE.
        wr(0,32'h100,2,0,2);if(starts!=0)$fatal(1,"disabled START accepted");
        wr(0,1,1,1,0);wr(0,32'h100,2,2,0);rd(0,1);
        if(starts!=1)$fatal(1,"START pulse count");
        wr(0,32'h301,3,0,2);
        wr(0,32'h4,1,0,2); // DONE is read-only.
        if(starts!=1||clears!=0)$fatal(1,"invalid CTRL command side effect");
        wr(0,32'h200,2,1,0);rd(0,1);
        if(clears!=1)$fatal(1,"CLEAR pulse count");
        // All actual writes, including ENABLE and CLEAR, are locked while BUSY.
        busy=1;done=1;error=1;error_code=4'ha;rd(0,32'haf);
        held_cfg={cfg_enable,cfg_src,cfg_dst,cfg_dim};
        wr(4,0,15,0,2);wr(8,0,15,1,2);wr(12,0,15,2,2);
        wr(0,32'h101,3,0,2);wr(0,32'h201,3,1,2);wr(0,0,1,2,2);
        wr(4,32'hffffffff,0,0,0);
        if({cfg_enable,cfg_src,cfg_dst,cfg_dim}!==held_cfg||starts!=1||clears!=1)
            $fatal(1,"BUSY protection failed");
        busy=0;done=0;error=0;error_code=0;
        // Protection is checked after AW and W, at the actual commit edge.
        begin_write(BASE+4,32'hdeadbeef,15,0);busy=1;finish_write(2);busy=0;
        rd(4,32'hddccbbaa);
        // Non-strobed command bits do not pulse START or CLEAR.
        wr(0,32'h301,1,0,0);
        if(starts!=1||clears!=1)$fatal(1,"command fired without byte-one strobe");
        wr(0,32'h101,3,0,0);
        if(starts!=2)$fatal(1,"combined ENABLE START failed");
        // Dynamic status cannot change an already accepted stalled R response.
        @(negedge clk);araddr=BASE;arvalid=1;
        @(posedge clk);while(!arready)@(posedge clk);
        @(negedge clk);arvalid=0;busy=1;done=1;error=1;error_code=15;
        repeat(4)begin @(posedge clk);#1;
            if(!rvalid||rdata!==1||rresp!==0)$fatal(1,"status read not snapshotted");
        end
        @(negedge clk);rready=1;@(posedge clk);@(negedge clk);rready=0;
        rd(0,32'hff);busy=0;done=0;error=0;error_code=0;
        // Reset discards an unmatched AW. A fresh W must wait for fresh AW.
        send_aw(BASE+4);
        @(negedge clk);rst_n=0;@(posedge clk);#1;
        if(bvalid||rvalid||start_pulse||clear_status_pulse||
           {cfg_enable,cfg_src,cfg_dst,cfg_dim}!==0)$fatal(1,"reset did not clear AXIL");
        @(negedge clk);rst_n=1;
        send_w(32'h89abcdef,15);
        repeat(4)begin @(posedge clk);#1;
            if(bvalid||cfg_src!==0||cfg_dst!==0)$fatal(1,"stale AW survived reset");
        end
        send_aw(BASE+8);finish_write(0);rd(4,0);rd(8,32'h89abcdef);
        // Reset also discards a W-only transaction and a held read response.
        send_w(32'h76543210,15);
        @(negedge clk);araddr=BASE+8;arvalid=1;
        @(posedge clk);while(!arready)@(posedge clk);
        @(negedge clk);arvalid=0;rst_n=0;
        @(posedge clk);#1;
        if(rvalid||bvalid||cfg_dst!==0)$fatal(1,"stalled-response reset failed");
        @(negedge clk);rst_n=1;
        send_aw(BASE+12);
        repeat(4)begin @(posedge clk);#1;
            if(rvalid||bvalid||cfg_dim!==0)$fatal(1,"stale W/R survived reset");
        end
        send_w(32'h0fff0002,15);finish_write(0);rd(12,32'h0fff0002);
        $display("PASS softmax AXI-Lite: decode, AW/W ordering, byte strobes, status snapshot, busy protection, reset");
        $finish;
    end
    initial begin #1000000;$fatal(1,"AXIL test timeout");end
endmodule
