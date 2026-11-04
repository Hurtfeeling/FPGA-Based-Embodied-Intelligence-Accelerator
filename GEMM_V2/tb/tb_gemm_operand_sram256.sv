`timescale 1ns/1ps
module tb_gemm_operand_sram256;
    reg clk=0;always #5 clk=~clk;
    reg we=0,ren=0;
    reg [13:0] waddr=0,raddr=0;
    reg [255:0] wdata=0;wire [255:0] rdata;
    gemm_bt_sram256 #(.DEPTH(16384),.ADDR_W(14),.PIPELINED_READ(1)) dut(.*);
    reg [255:0] model[0:16383],p0,p1,p2,expected;
    reg v0=0,v1=0,v2=0,known=0;
    reg prev_ren=0;reg [13:0] prev_addr=0;
    integer checks=0,cycle=0,seed=27183,i,b,k,count;
    wire [7:0] slice_ren;
    genvar g;
    generate for(g=0;g<8;g=g+1)begin:monitor
        assign slice_ren[g]=dut.g_sliced.g_slice[g].read_en_q;
    end endgenerate
    function automatic [255:0] pattern(input integer addr,id);
        integer j;begin for(j=0;j<8;j=j+1)pattern[j*32+:32]=32'hf17c8329^(addr*193+id*59+j*137);end
    endfunction
    function automatic [13:0] address(input integer id);
        begin
            case(id%4)
                0:address=(id/4)*2048;
                1:address=(id/4)*2048+1;
                2:address=(id/4)*2048+2046;
                3:address=(id/4)*2048+2047;
            endcase
        end
    endfunction
    always @(posedge clk)begin
        cycle=cycle+1;
        p2=p1;p1=p0;v2=v1;v1=v0;v0=ren;
        if(ren)p0=model[raddr];
        if(we)begin
            if((ren&&waddr==raddr)||(prev_ren&&waddr==prev_addr))$fatal(1,"test read/write collision");
            model[waddr]=wdata;
        end
        prev_ren=ren;prev_addr=raddr;
        if(v2)begin expected=p2;known=1;end
        #1;
        if(known)begin
            if(rdata!==expected)$fatal(1,"sliced SRAM return/tag/hold mismatch cycle=%0d",cycle);
            checks=checks+1;
        end
        count=0;for(k=0;k<8;k=k+1)count=count+slice_ren[k];
        if(count!=(ren?1:0))$fatal(1,"more than one read slice activated");
    end
    initial begin
        repeat(3)@(negedge clk);
        for(i=0;i<32;i=i+1)begin
            @(negedge clk);we=1;waddr=address(i);wdata=pattern(waddr,i);
        end
        @(negedge clk);we=0;
        // Consecutive requests cross every 64KiB boundary and the last address.
        for(i=0;i<64;i=i+1)begin @(negedge clk);ren=1;raddr=address(i%32);end
        for(i=0;i<1600;i=i+1)begin
            @(negedge clk);ren=($random(seed)&3)!=0;raddr=address($random(seed)&31);
            we=($random(seed)&3)!=0;waddr=address($random(seed)&31);
            while((ren&&waddr==raddr)||(prev_ren&&waddr==prev_addr))waddr=address($random(seed)&31);
            wdata=pattern(waddr,i+32);
        end
        @(negedge clk);we=0;ren=0;repeat(8)@(negedge clk);
        if(checks<1600)$fatal(1,"sliced memory coverage insufficient");
        $display("PASS operand SRAM: 8x64KiB, one256bit read/clock, latency3, all slice boundaries/last address, independent R/W and hold");$finish;
    end
    initial begin #100000;$fatal(1,"operand SRAM timeout");end
endmodule
