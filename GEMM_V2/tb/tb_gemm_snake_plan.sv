`timescale 1ns/1ps
// 仅运行启动规划阶段，独立枚举四条路径；不请求 DDR、不复制 DUT 成本公式。
module tb_gemm_snake_plan;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,start_pulse=0,clear_status_pulse=0;
    reg [191:0] cfg_flat=0;
    wire [479:0] ro_flat;
    wire busy,irq,dma_cmd_valid,dma_rx_ready,dma_tx_valid,wr_cmd_valid;
    wire [31:0] dma_cmd_addr,dma_cmd_bytes,wr_cmd_addr,wr_cmd_bytes;
    wire [127:0] dma_tx_data;
    gemm_core dut(.clk(clk),.rst_n(rst_n),.cfg_flat(cfg_flat),.start_pulse(start_pulse),
        .clear_status_pulse(clear_status_pulse),.busy(busy),.irq(irq),.ro_flat(ro_flat),
        .dma_cmd_valid(dma_cmd_valid),.dma_cmd_ready(1'b0),.dma_cmd_addr(dma_cmd_addr),
        .dma_cmd_bytes(dma_cmd_bytes),.dma_done(1'b0),.dma_error(1'b0),.dma_busy(1'b0),
        .dma_rx_data(128'd0),.dma_rx_valid(1'b0),.dma_rx_ready(dma_rx_ready),
        .dma_tx_data(dma_tx_data),.dma_tx_valid(dma_tx_valid),.dma_tx_ready(1'b0),
        .wr_cmd_valid(wr_cmd_valid),.wr_cmd_ready(1'b0),.wr_cmd_addr(wr_cmd_addr),
        .wr_cmd_bytes(wr_cmd_bytes),.wr_done(1'b0),.wr_error(1'b0),.wr_busy(1'b0),
        .bus_read_beat(1'b0),.bus_write_beat(1'b0));
    integer tests=0;
    task automatic check_plan(input integer m,n,k);
        integer cap,tm,tn,cm,cn,candidate,o,i,j,outer,inner,mr,nr,pm,pn,best,timeout;
        reg [63:0] cost,best_cost;
        begin
            cap=(524288/(32*k))*32;if(cap>256)cap=256;
            tm=(m<cap)?m:cap;tn=(n<cap)?n:cap;
            cm=(m+tm-1)/tm;cn=(n+tn-1)/tn;
            best=0;best_cost=64'hffffffffffffffff;
            for(candidate=0;candidate<4;candidate=candidate+1)begin
                outer=(candidate/2)?cn:cm;inner=(candidate/2)?cm:cn;
                cost=0;pm=-1;pn=-1;
                for(o=0;o<outer;o=o+1)for(i=0;i<inner;i=i+1)begin
                    j=((candidate%2)^(o%2))?(inner-1-i):i;
                    mr=(candidate/2)?j*tm:o*tm;nr=(candidate/2)?o*tn:j*tn;
                    if(mr!=pm)cost=cost+((m-mr<tm)?m-mr:tm);
                    if(nr!=pn)cost=cost+((n-nr<tn)?n-nr:tn);
                    pm=mr;pn=nr;
                end
                if(cost<best_cost)begin best=candidate;best_cost=cost;end
            end
            @(negedge clk);rst_n=0;start_pulse=0;
            @(negedge clk);rst_n=1;
            cfg_flat=0;cfg_flat[32+:32]=32'h10000;cfg_flat[64+:32]=32'h200000;
            cfg_flat[96+:32]=32'h400000;cfg_flat[128+:32]=(m/32)|((n/32)<<16);
            cfg_flat[160+:32]=k/32;start_pulse=1;
            @(negedge clk);start_pulse=0;timeout=0;
            while(dut.state!=dut.TILE_SETUP)begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>160||dut.error_q)$fatal(1,"planner stalled/error M=%0d N=%0d K=%0d",m,n,k);
            end
            if(dut.tile_m!=tm||dut.tile_n!=tn||dut.snake_columns!=best/2||dut.snake_reverse!=best%2||
               dut.m0!=((best/2&&best%2)?(cm-1)*tm:0)||
               dut.n0!=((! (best/2)&&best%2)?(cn-1)*tn:0))
                $fatal(1,"planner mismatch M=%0d N=%0d K=%0d wantRoute=%0d actual=%0d/%0d start=%0d/%0d",m,n,k,
                    best,dut.snake_columns,dut.snake_reverse,dut.m0,dut.n0);
            tests=tests+1;
        end
    endtask
    integer kvals[0:15];
    integer ii,jj,mm,nn;
    initial begin
        kvals[0]=32;kvals[1]=2048;kvals[2]=2080;kvals[3]=2336;
        kvals[4]=2368;kvals[5]=2720;kvals[6]=2752;kvals[7]=3264;
        kvals[8]=3296;kvals[9]=4096;kvals[10]=4128;kvals[11]=5440;
        kvals[12]=5472;kvals[13]=8192;kvals[14]=8224;kvals[15]=16352;
        for(ii=0;ii<16;ii=ii+1)begin
            check_plan(32,32,kvals[ii]);check_plan(288,288,kvals[ii]);
            check_plan(288,768,kvals[ii]);check_plan(768,288,kvals[ii]);
            check_plan(288,512,kvals[ii]);check_plan(512,288,kvals[ii]);
            check_plan(544,800,kvals[ii]);check_plan(800,544,kvals[ii]);
            for(jj=0;jj<4;jj=jj+1)begin
                mm=32*(1+(ii*13+jj*7)%32);nn=32*(1+(ii*5+jj*11)%32);
                check_plan(mm,nn,kvals[ii]);
            end
        end
        $display("PASS snake planner independent path enumeration cases=%0d all K capacity boundaries",tests);
        $finish;
    end
    initial begin #10000000;$fatal(1,"planner global timeout");end
endmodule
