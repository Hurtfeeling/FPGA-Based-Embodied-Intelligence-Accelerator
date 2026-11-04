`timescale 1ns/1ps
// 检查混合缓存的真实边界、每拍读取、独立写端口、3 拍延迟与输出保持。
module tb_gemm_mixed_sram256;
    reg clk=0;always #5 clk=~clk;
    reg we=0,ren=0;
    reg [13:0] waddr=0,raddr=0;
    reg [255:0] wdata=0;wire [255:0] rdata;
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1),.MIXED_RAM(1)) dut(.*);
    reg [255:0] model[0:16383],p0,p1,p2,expected;
    reg v0=0,v1=0,v2=0,known=0;
    reg prev_ren=0;reg [13:0] prev_addr=0;
    integer checks=0,cycle=0,seed=27183,i,k,count;
    wire [5:0] bank_ren;
    genvar g;
    generate for(g=0;g<4;g=g+1)begin:monitor_bram
        assign bank_ren[g]=dut.g_mixed.g_bram[g].read_en_q;
    end
    for(g=0;g<2;g=g+1)begin:monitor_uram
        assign bank_ren[4+g]=dut.g_mixed.g_uram[g].read_en_q;
    end endgenerate
    function automatic [255:0] pattern(input integer addr,id);
        integer j;begin for(j=0;j<8;j=j+1)pattern[j*32+:32]=32'hf17c8329^(addr*193+id*59+j*137);end
    endfunction
    function automatic [13:0] address(input integer id);
        begin
            // 同时覆盖四个 BRAM bank 与两个 URAM bank 的首尾，以及 8191/8192 分界。
            case(id%24)
                0:address=0;1:address=1;2:address=2046;3:address=2047;
                4:address=2048;5:address=2049;6:address=4094;7:address=4095;
                8:address=4096;9:address=4097;10:address=6142;11:address=6143;
                12:address=6144;13:address=6145;14:address=8190;15:address=8191;
                16:address=8192;17:address=8193;18:address=12286;19:address=12287;
                20:address=12288;21:address=12289;22:address=16382;23:address=16383;
            endcase
        end
    endfunction
    always @(posedge clk)begin
        cycle=cycle+1;
        p2=p1;p1=p0;v2=v1;v1=v0;v0=ren;
        if(ren)p0=model[raddr];
        if(we)begin
            // RAM 实际读发生在请求后一拍，避免跨周期同址冲突的不确定语义。
            if((ren&&waddr==raddr)||(prev_ren&&waddr==prev_addr))$fatal(1,"test read/write collision");
            model[waddr]=wdata;
        end
        prev_ren=ren;prev_addr=raddr;
        if(v2)begin expected=p2;known=1;end
        #1;
        if(known)begin
            if(rdata!==expected)$fatal(1,"mixed SRAM latency/tag/hold mismatch cycle=%0d addr=%0d",cycle,raddr);
            checks=checks+1;
        end
        count=0;for(k=0;k<6;k=k+1)count=count+bank_ren[k];
        if(count!=(ren?1:0))$fatal(1,"mixed SRAM activates more than one bank");
    end
    initial begin
        repeat(3)@(negedge clk);
        for(i=0;i<24;i=i+1)begin
            @(negedge clk);we=1;waddr=address(i);wdata=pattern(waddr,i);
        end
        @(negedge clk);we=0;
        // 交替访问 BRAM 与 URAM，验证两种存储返回延迟完全一致。
        for(i=0;i<96;i=i+1)begin @(negedge clk);ren=1;raddr=address((i%2)?16+(i/2)%8:(i/2)%16);end
        for(i=0;i<1600;i=i+1)begin
            @(negedge clk);ren=($random(seed)&3)!=0;raddr=address(($random(seed)&31)%24);
            we=($random(seed)&3)!=0;waddr=address(($random(seed)&31)%24);
            while((ren&&waddr==raddr)||(prev_ren&&waddr==prev_addr))waddr=address(($random(seed)&31)%24);
            wdata=pattern(waddr,i+24);
        end
        @(negedge clk);we=0;ren=0;repeat(8)@(negedge clk);
        if(checks<1600)$fatal(1,"mixed SRAM coverage insufficient");
        $display("PASS mixed operand SRAM: 256KiB BRAM +256KiB URAM, latency3, boundaries, independent R/W and hold");$finish;
    end
    initial begin #100000;$fatal(1,"mixed SRAM timeout");end
endmodule
