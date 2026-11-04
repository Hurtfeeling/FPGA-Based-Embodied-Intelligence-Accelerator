`timescale 1ns/1ps
module tb_gemm_job_status;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    reg [191:0] cfg_flat=0;wire [479:0] ro_flat;
    reg start_pulse=0,clear_status_pulse=0;wire busy,irq;
    wire dma_cmd_valid;reg dma_cmd_ready=0;wire [31:0] dma_cmd_addr,dma_cmd_bytes;
    reg dma_done=0,dma_error=0,dma_busy=0;wire dma_rx_ready,dma_tx_valid;
    wire [127:0] dma_tx_data;wire wr_cmd_valid;wire [31:0] wr_cmd_addr,wr_cmd_bytes;
    reg wr_busy=0,bus_read_beat=0,bus_write_beat=0;
    gemm_core dut(.clk(clk),.rst_n(rst_n),.cfg_flat(cfg_flat),.start_pulse(start_pulse),.clear_status_pulse(clear_status_pulse),
        .busy(busy),.irq(irq),.ro_flat(ro_flat),.dma_cmd_valid(dma_cmd_valid),.dma_cmd_ready(dma_cmd_ready),
        .dma_cmd_addr(dma_cmd_addr),.dma_cmd_bytes(dma_cmd_bytes),.dma_done(dma_done),.dma_error(dma_error),.dma_busy(dma_busy),
        .dma_rx_data(128'd0),.dma_rx_valid(1'b0),.dma_rx_ready(dma_rx_ready),
        .dma_tx_data(dma_tx_data),.dma_tx_valid(dma_tx_valid),.dma_tx_ready(1'b0),
        .wr_cmd_valid(wr_cmd_valid),.wr_cmd_ready(1'b0),.wr_cmd_addr(wr_cmd_addr),.wr_cmd_bytes(wr_cmd_bytes),
        .wr_done(1'b0),.wr_error(1'b0),.wr_busy(wr_busy),.bus_read_beat(bus_read_beat),.bus_write_beat(bus_write_beat));
    task automatic reset_core;
        begin
            @(negedge clk);rst_n=0;start_pulse=0;clear_status_pulse=0;dma_done=0;dma_error=0;dma_busy=0;dma_cmd_ready=0;wr_busy=0;
            @(posedge clk);@(negedge clk);rst_n=1;
        end
    endtask
    task automatic start_job;
        begin @(negedge clk);start_pulse=1;@(posedge clk);@(negedge clk);start_pulse=0;end
    endtask
    task automatic clear_flags;
        begin @(negedge clk);clear_status_pulse=1;@(posedge clk);@(negedge clk);clear_status_pulse=0;end
    endtask
    integer k,cap,tm,tk,timeout;
    reg [31:0] previous_job;
    task automatic actual_tile(input integer kval);
        begin
            reset_core();cfg_flat[160+:32]=kval/32;start_job();timeout=0;
            while(ro_flat[11:8]==0)begin @(negedge clk);timeout=timeout+1;if(timeout>100)$fatal(1,"auto config stalled");end
            tk=kval;cap=(524288/(32*tk))*32;tm=(cap<256)?cap:256;
            if(ro_flat[11:8]!==tm/32||ro_flat[15:12]!==tm/32||ro_flat[25:16]!==tk/32)
                $fatal(1,"wrong automatic tile K=%0d FLAGS=%h",kval,ro_flat[31:0]);
        end
    endtask
    task automatic span_case(input [31:0] aa,bb,yy,mn,kk,input bit bad);
        begin
            reset_core();cfg_flat[32+:32]=aa;cfg_flat[64+:32]=bb;cfg_flat[96+:32]=yy;
            cfg_flat[128+:32]=mn;cfg_flat[160+:32]=kk;start_job();timeout=0;
            while(!ro_flat[2]&&!dma_cmd_valid)begin
                @(negedge clk);timeout=timeout+1;if(timeout>120)$fatal(1,"serial span validation stalled");
            end
            if(ro_flat[2]!==bad || (bad&&dma_cmd_valid))$fatal(1,"span check mismatch A%h BT%h Y%h MN%h K%h bad%0d",aa,bb,yy,mn,kk,bad);
        end
    endtask
    // Boundary events are safe even with a coincident fault. A fault while
    // computing must still retain the accumulator until accepted DMA drains.
    // Seed one accumulator so the checks cannot pass on an already-zero PE.
    task automatic clear_boundary_case(input integer boundary);
        begin
            reset_core();
            @(negedge clk);
            dut.task_active=1;
            if(boundary==0)dut.state=dut.SA_CLEAR;
            else if(boundary==1)begin
                dut.state=dut.WAIT_TILE;
                force dut.micro_done=1'b1;
            end else dut.state=dut.FEED;
            dut.u_array.g_row[0].g_pe[0].u_mac.accum_q=32'sd12345;
            dma_busy=1;wr_busy=1;dma_done=1;dma_error=1;
            #1;
            if(!dut.abort_all||dut.sa_clear!==(boundary!=2))
                $fatal(1,"clear/fault boundary decode failed boundary=%0d",boundary);
            @(posedge clk);#1;
            if(!ro_flat[2]||!busy||dut.store_fire||dma_cmd_valid||wr_cmd_valid)
                $fatal(1,"clear optimization changed fault suppression");
            if(dut.u_array.accum[0]!==((boundary==2)?32'd12345:32'd0))
                $fatal(1,"unsafe or missing PE clear boundary=%0d",boundary);
            @(negedge clk);
            if(boundary==1)release dut.micro_done;
            dma_done=0;dma_error=0;
            // An outstanding read OR write must keep fault cleanup waiting.
            dut.u_array.g_row[0].g_pe[0].u_mac.accum_q=32'sd6789;
            repeat(2)begin
                @(posedge clk);#1;
                if(dut.sa_clear||dut.u_array.accum[0]!==32'd6789||!busy)
                    $fatal(1,"PE cleared before accepted DMA drained");
            end
            @(negedge clk);dma_busy=0;
            @(posedge clk);#1;
            if(dut.sa_clear||dut.u_array.accum[0]!==32'd6789||!busy)
                $fatal(1,"PE cleared while write DMA was busy");
            @(negedge clk);dma_busy=1;wr_busy=0;
            @(posedge clk);#1;
            if(dut.sa_clear||dut.u_array.accum[0]!==32'd6789||!busy)
                $fatal(1,"PE cleared while read DMA was busy");
            @(negedge clk);dma_busy=0;
            #1;if(!dut.sa_clear)$fatal(1,"missing drained-fault clear");
            @(posedge clk);#1;
            if(busy||dut.u_array.accum[0]!==0||!ro_flat[2])
                $fatal(1,"drained fault did not clear PE and retain ERROR");
        end
    endtask
    initial begin
        cfg_flat[0+:32]=3;cfg_flat[32+:32]=32'h10000;cfg_flat[64+:32]=32'h30000000;cfg_flat[96+:32]=32'h60000000;
        cfg_flat[128+:32]=32'h00800080;
        for(k=32;k<=16352;k=k+32)begin
            tk=k;cap=(524288/(32*tk))*32;tm=(cap<256)?cap:256;
            if(dut.max_rows(tk)!==tm)$fatal(1,"capacity lookup mismatch K%0d",k);
        end
        actual_tile(32);actual_tile(96);actual_tile(2048);actual_tile(2336);actual_tile(2720);actual_tile(3072);
        actual_tile(4096);actual_tile(5440);actual_tile(8192);actual_tile(12288);actual_tile(16320);actual_tile(16352);
        // End-exclusive == 2^32 is legal; one aligned step beyond is rejected.
        span_case(32'hfffffc00,32'h10000,32'h20000,32'h00010001,1,0);
        span_case(32'hfffffc20,32'h10000,32'h20000,32'h00010001,1,1);
        span_case(32'h10000,32'hfffffc00,32'h20000,32'h00010001,1,0);
        span_case(32'h10000,32'hfffffc20,32'h20000,32'h00010001,1,1);
        span_case(32'h10000,32'h20000,32'hfffff000,32'h00010001,1,0);
        span_case(32'h10000,32'h20000,32'hfffff020,32'h00010001,1,1);
        span_case(0,0,0,32'hffffffff,1,1); // Large Y product cannot be truncated.
        span_case(32'hfff80400,32'h10000,32'h20000,32'h00010001,511,0);
        span_case(32'hfff80420,32'h10000,32'h20000,32'h00010001,511,1);
        span_case(0,0,0,32'h00010001,512,1);
        if(ro_flat[25:8]!=0)$fatal(1,"invalid K published automatic dimensions");
        span_case(0,0,0,32'h00010001,513,1);
        cfg_flat[32+:32]=32'h10000;cfg_flat[64+:32]=32'h30000000;cfg_flat[96+:32]=32'h60000000;
        // Real read error while another write-side busy keeps fault cleanup active.
        reset_core();cfg_flat[128+:32]=32'h00010001;cfg_flat[160+:32]=2;start_job();
        while(!dma_cmd_valid)@(negedge clk);
        dma_cmd_ready=1;dma_busy=1;@(posedge clk);@(negedge clk);
        dma_cmd_ready=0;dma_busy=0;dma_done=1;dma_error=1;wr_busy=1;
        @(posedge clk);@(negedge clk);dma_done=0;dma_error=0;
        if(!ro_flat[2]||!busy||!irq)$fatal(1,"global ERROR not latched");
        previous_job=ro_flat[64+:32];clear_flags();start_job();
        if(!ro_flat[2]||ro_flat[64+:32]!==previous_job)$fatal(1,"busy clear/start removed ERROR");
        wr_busy=0;timeout=0;while(busy)begin @(negedge clk);timeout=timeout+1;if(timeout>20)$fatal(1,"fault did not drain");end
        clear_flags();if(ro_flat[2])$fatal(1,"idle ERROR clear failed");
        // Set must win a coincident CLEAR.
        cfg_flat[128+:32]=0;start_job();clear_status_pulse=1;
        @(posedge clk);@(negedge clk);clear_status_pulse=0;
        if(!ro_flat[2])$fatal(1,"CLEAR overrode a new error");
        while(busy)@(negedge clk);clear_flags();
        // Counter saturation and CLEAR+new traffic count from zero.
        dut.read_bytes_q=32'hfffffff0;bus_read_beat=1;@(posedge clk);@(negedge clk);bus_read_beat=0;
        if(ro_flat[384+:32]!==32'hffffffff||!ro_flat[4])$fatal(1,"counter saturation failed");
        bus_read_beat=1;clear_flags();bus_read_beat=0;
        if(ro_flat[384+:32]!==16||ro_flat[4])$fatal(1,"clear/new traffic priority failed");
        if(ro_flat[3]||ro_flat[5]||ro_flat[448+:32]!==32'h0a080020)$fatal(1,"removed mode or CAP mismatch");
        clear_boundary_case(0);clear_boundary_case(1);clear_boundary_case(2);
        $display("PASS auto tiles/status: K lookup, spans, counters, coincident clear/fault boundaries and DMA drain-before-clear");$finish;
    end
    initial begin #1000000;$fatal(1,"auto/status timeout");end
endmodule
