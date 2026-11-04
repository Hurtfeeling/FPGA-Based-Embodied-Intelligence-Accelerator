`timescale 1ns/1ps
module result_ram_case #(parameter STYLE="block")(output reg done);
    reg clk=0;always #5 clk=~clk;
    reg we=0,ren=0;reg [11:0] waddr=0,raddr=0;
    reg [511:0] wdata=0;wire [511:0] rdata;
    reg [511:0] model[0:4095],pipe0,pipe1,expected;
    reg v0=0,v1=0,known=0;
    integer i,seed=316,checks=0;
    gemm_result_sram512 #(.RAM_STYLE(STYLE)) dut(.*);
    function automatic [511:0] data(input integer id);
        integer j;begin for(j=0;j<16;j=j+1)data[j*32+:32]=32'haf813295^(id*137+j*59);end
    endfunction
    always @(posedge clk)begin
        pipe1=pipe0;v1=v0;v0=ren;
        if(ren)pipe0=model[raddr];
        if(we)begin
            if(ren&&waddr==raddr)$fatal(1,"test generated a memory collision");
            model[waddr]=wdata;
        end
        if(v1)begin expected=pipe1;known=1;end
        #1;
        if(known&&rdata!==expected)$fatal(1,"512-bit result SRAM latency/hold mismatch");
        if(known)checks=checks+1;
    end
    initial begin
        done=0;
        for(i=0;i<32;i=i+1)begin
            @(negedge clk);we=1;waddr=(i<16)?i:4080+i-16;wdata=data(i);
        end
        @(negedge clk);we=0;
        for(i=0;i<600;i=i+1)begin
            @(negedge clk);we=$random(seed)&1;ren=($random(seed)&3)!=0;
            waddr=($random(seed)&15)+(($random(seed)&1)?4080:0);
            raddr=($random(seed)&15)+(($random(seed)&1)?4080:0);
            if(we&&waddr==raddr)raddr=(raddr&12'hff0)|((raddr+1)&15);
            wdata=data(i+32);
        end
        @(negedge clk);we=0;ren=0;repeat(4)@(posedge clk);#2;
        if(checks<500)$fatal(1,"result RAM coverage insufficient");
        $display("PASS result RAM512 %s: first/last addresses, two-cycle read, hold, independent ports",STYLE);done=1;
    end
endmodule
module tb_gemm_result_sram512;
    wire d0,d1;result_ram_case #(.STYLE("block")) t0(d0);result_ram_case #(.STYLE("ultra")) t1(d1);
    initial begin wait(d0&&d1);$display("PASS tb_gemm_result_sram512");$finish;end
    initial begin #100000;$fatal(1,"result RAM timeout");end
endmodule
