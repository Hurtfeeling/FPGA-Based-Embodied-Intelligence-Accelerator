`timescale 1ns/1ps
module tb_gemm_axil_regs;
    localparam [31:0] BASE=32'ha0000000;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,busy=0;
    wire [191:0] cfg_flat;reg [479:0] ro_flat=0;
    wire start_pulse,clear_status_pulse;
    reg [31:0] awaddr=0,wdata=0,araddr=0;
    reg [3:0] wstrb=0;
    reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
    wire awready,wready,bvalid,arready,rvalid;
    wire [1:0] bresp,rresp;wire [31:0] rdata;
    integer starts=0,clears=0,requests=0,i,expected_starts,expected_clears;
    reg [31:0] held_word;reg [191:0] held_cfg;
    gemm_axil_regs #(.C_S_AXI_BASEADDR(BASE)) dut(
        .busy(busy),.cfg_flat(cfg_flat),.ro_flat(ro_flat),
        .start_pulse(start_pulse),.clear_status_pulse(clear_status_pulse),
        .S_AXI_ACLK(clk),.S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr),.S_AXI_AWPROT(3'd0),.S_AXI_AWVALID(awvalid),.S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),.S_AXI_WSTRB(wstrb),.S_AXI_WVALID(wvalid),.S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),.S_AXI_BVALID(bvalid),.S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),.S_AXI_ARPROT(3'd0),.S_AXI_ARVALID(arvalid),.S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),.S_AXI_RRESP(rresp),.S_AXI_RVALID(rvalid),.S_AXI_RREADY(rready));
    always @(posedge clk)if(!rst_n)begin starts=0;clears=0;end
        else begin
            if(start_pulse)starts=starts+1;if(clear_status_pulse)clears=clears+1;
        end

    task automatic send_aw(input [31:0] address);
        begin @(negedge clk);awaddr=address;awvalid=1;
            @(posedge clk);while(!awready)@(posedge clk);
            @(negedge clk);awvalid=0;end
    endtask
    task automatic send_w(input [31:0] data,input [3:0] strobe);
        begin @(negedge clk);wdata=data;wstrb=strobe;wvalid=1;
            @(posedge clk);while(!wready)@(posedge clk);
            @(negedge clk);wvalid=0;end
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
        begin timeout=0;
            while(!bvalid&&timeout<60)begin @(posedge clk);#1;timeout=timeout+1;end
            if(!bvalid||bresp!==expected_resp)$fatal(1,"write response addr=%h got=%b expected=%b",awaddr,bresp,expected_resp);
            held=bresp;
            repeat(2)begin @(posedge clk);#1;
                if(!bvalid||bresp!==held||awready||wready)$fatal(1,"write response backpressure failed");end
            @(negedge clk);bready=1;@(posedge clk);@(negedge clk);bready=0;
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
                $fatal(1,"read addr=%h got=%h/%b expected=%h/%b",address,rdata,rresp,expected,response);
            held=rdata;held_resp=rresp;
            repeat(2)begin @(posedge clk);#1;
                if(!rvalid||rdata!==held||rresp!==held_resp||arready)$fatal(1,"read response backpressure failed");end
            @(negedge clk);rready=1;@(posedge clk);@(negedge clk);rready=0;
        end
    endtask
    task automatic rd(input [31:0] offset,input [31:0] expected);
        begin rd_abs(BASE+offset,expected,0);end
    endtask

    initial begin
        for(i=0;i<15;i=i+1)ro_flat[i*32+:32]=32'hc0700000+i;
        ro_flat[2]=0;
        repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
        for(i=0;i<6;i=i+1)rd(i*4,0);
        for(i=1;i<5;i=i+1)begin wr(i*4,32'h12340000+i,15,i%3,0);rd(i*4,32'h12340000+i);end
        wr(32'h14,32'h1234,15,2,0);rd(32'h14,32'h1234);
        wr(32'h14,32'h10000,15,0,2);rd(32'h14,32'h1234);
        wr(4,32'h11223344,15,0,0);wr(4,32'haa,1,1,0);wr(4,32'hbb00,2,2,0);
        wr(4,32'hcc0000,4,0,0);wr(4,32'hdd000000,8,1,0);rd(4,32'hddccbbaa);
        for(i=0;i<15;i=i+1)begin rd(32'h18+i*4,ro_flat[i*32+:32]);wr(32'h18+i*4,0,15,i%3,2);end
        write_abs(BASE-4,0,15,0,2);write_abs(BASE+32'h10000004,0,15,1,2);
        wr(32'h54,0,15,2,2);wr(2,0,15,0,2);
        rd_abs(BASE+32'h54,0,2);rd_abs(BASE+32'h10000004,0,2);rd_abs(BASE-4,0,2);rd_abs(BASE+2,0,2);
        // No side effects from zero strobe, even at an unmapped address.
        held_cfg=cfg_flat;write_abs(32'hffffffff,32'hffffffff,0,2,0);
        if(cfg_flat!==held_cfg)$fatal(1,"zero-strobe side effect");
        wr(0,32'h101,3,1,0);rd(0,1);if(starts!=1)$fatal(1,"combined ENABLE START");
        wr(0,32'h301,3,2,2);if(starts!=1||clears!=0)$fatal(1,"mixed START CLEAR accepted");
        busy=1;wr(4,0,15,0,2);rd(4,32'hddccbbaa);
        wr(0,32'h103,3,0,2);rd(0,1);wr(0,32'h203,3,2,0);rd(0,3);
        if(starts!=1||clears!=1)$fatal(1,"busy protection failed");
        busy=0;ro_flat[2]=1;wr(0,32'h101,3,1,2);rd(0,3);
        if(starts!=1)$fatal(1,"ERROR START accepted");
        ro_flat[2]=0;wr(0,0,1,2,0);wr(0,32'h100,2,0,2);rd(0,0);
        wr(0,32'h101,3,1,0);if(starts!=2)$fatal(1,"restart failed");
        // Pipeline must re-check protection at commit, after AW/W acceptance.
        held_cfg=cfg_flat;begin_write(BASE+4,32'hdeadbeef,15,0);busy=1;
        finish_write(2);busy=0;if(cfg_flat!==held_cfg)$fatal(1,"late BUSY write accepted");
        begin_write(BASE,32'h101,3,0);ro_flat[2]=1;finish_write(2);ro_flat[2]=0;
        if(starts!=2)$fatal(1,"late ERROR START accepted");
        // A reset between AW and W must drop the old address. A fresh W
        // alone cannot commit until a fresh AW arrives, regardless of payload.
        send_aw(BASE+4);
        @(negedge clk);rst_n=0;
        @(posedge clk);#1;
        if(bvalid||rvalid||start_pulse||clear_status_pulse||cfg_flat!==0)
            $fatal(1,"AXIL reset did not invalidate pending transaction");
        @(negedge clk);rst_n=1;
        send_w(32'h89abcdef,15);
        repeat(4)begin @(posedge clk);#1;if(bvalid||cfg_flat!==0)$fatal(1,"old AW committed after reset");end
        send_aw(BASE+8);finish_write(0);
        rd(4,0);rd(8,32'h89abcdef);
        // Drop a stalled R response and a W-only transaction, then verify
        // that neither old response nor old write data can reappear.
        send_w(32'h76543210,15);
        @(negedge clk);araddr=BASE+8;arvalid=1;
        @(posedge clk);while(!arready)@(posedge clk);
        @(negedge clk);arvalid=0;
        wait(rvalid);
        @(negedge clk);rst_n=0;
        @(posedge clk);#1;
        if(rvalid||bvalid||cfg_flat!==0)$fatal(1,"AXIL stalled response reset failed");
        @(negedge clk);rst_n=1;
        send_aw(BASE+12);
        repeat(4)begin @(posedge clk);#1;if(bvalid||rvalid||cfg_flat!==0)$fatal(1,"old W/R reappeared after reset");end
        send_w(32'h13579bdf,15);finish_write(0);
        rd(8,0);rd(12,32'h13579bdf);
        $display("PASS ABI10 AXI-Lite map, absolute decode, independent AW/W, WSTRB, backpressure and ERROR guard");$finish;
    end
    initial begin #1000000;$fatal(1,"AXIL timeout");end
endmodule
