`timescale 1ns/1ps
module tb_gemm_input_stream256 #(parameter integer READ_LATENCY=1);
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,cancel=0,start=0,advance=0;
    reg [8:0] k_words=1;
    reg [13:0] a_base=0,bt_base=0;
    wire read_valid;
    wire [13:0] a_addr,bt_addr;
    wire [255:0] a_data,bt_data;
    wire [255:0] a_vec,bt_vec;
    wire [31:0] a_valid,bt_valid;
    gemm_input_stream256 #(.READ_LATENCY(READ_LATENCY)) dut(.*);
    function automatic [7:0] value(input integer addr,byte_id,salt);
        value=(addr*37+byte_id*13+salt)&255;
    endfunction
    integer b,lp;
    reg [255:0] a_ram_pipe[0:READ_LATENCY-1],bt_ram_pipe[0:READ_LATENCY-1];
    assign a_data=a_ram_pipe[READ_LATENCY-1];
    assign bt_data=bt_ram_pipe[READ_LATENCY-1];
    // Explicit RAM return latency; verify tags follow all pipeline stages.
    always @(posedge clk)begin
        if(read_valid)for(b=0;b<32;b=b+1)begin
            a_ram_pipe[0][b*8+:8]<=value(a_addr,b,17);
            bt_ram_pipe[0][b*8+:8]<=value(bt_addr,b,91);
        end
        for(lp=1;lp<READ_LATENCY;lp=lp+1)begin
            a_ram_pipe[lp]<=a_ram_pipe[lp-1];bt_ram_pipe[lp]<=bt_ram_pipe[lp-1];
        end
    end
    integer request_count=0;
    always @(posedge clk)begin
        if(!rst_n||clear||cancel||start)request_count=0;
        else if(read_valid)begin
            if(a_addr!=a_base+(request_count%32)*k_words+request_count/32||
               bt_addr!=bt_base+(request_count%32)*k_words+request_count/32)
                $fatal(1,"round-robin address mismatch request=%0d",request_count);
            request_count=request_count+1;
        end
    end
    task automatic launch(input integer kval,abase,bbase);
        begin
            @(negedge clk);k_words=kval/32;a_base=abase;bt_base=bbase;start=1;
            @(posedge clk);@(negedge clk);start=0;
            repeat(READ_LATENCY+1)@(posedge clk);
            @(negedge clk);advance=1;
        end
    endtask
    task automatic run(input integer kval,abase,bbase);
        integer step,lane,kval_lane,addr_a,addr_b;
        begin
            launch(kval,abase,bbase);
            for(step=0;step<kval+67;step=step+1)begin
                @(posedge clk);
                for(lane=0;lane<32;lane=lane+1)begin
                    kval_lane=step-lane;
                    if(a_valid[lane]!==((kval_lane>=0)&&(kval_lane<kval))||bt_valid[lane]!==a_valid[lane])
                        $fatal(1,"lane valid/skew K=%0d step=%0d lane=%0d",kval,step,lane);
                    if(kval_lane>=0&&kval_lane<kval)begin
                        addr_a=abase+lane*(kval/32)+kval_lane/32;
                        addr_b=bbase+lane*(kval/32)+kval_lane/32;
                        if(a_vec[lane*8+:8]!==value(addr_a,kval_lane%32,17)||
                           bt_vec[lane*8+:8]!==value(addr_b,kval_lane%32,91))
                            $fatal(1,"lane refill/byte order K=%0d step=%0d lane=%0d",kval,step,lane);
                    end
                end
                @(negedge clk);
            end
            advance=0;
            if(read_valid||a_valid||bt_valid||request_count!=kval)$fatal(1,"input stream did not drain");
            @(negedge clk);clear=1;@(posedge clk);@(negedge clk);clear=0;
            $display("PASS serial256 input K=%0d, bases=%0d/%0d, exact K reads, all lane bytes/valid checked",kval,abase,bbase);
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        run(32,0,4096);run(64,128,512);run(96,17,8096);
        run(2048,14336,12288);run(8192,8192,0);run(16352,0,0);
        launch(96,1024,2048);repeat(19)@(negedge clk);cancel=1;
        @(posedge clk);#1;if(a_valid||bt_valid||read_valid)$fatal(1,"cancel left input tokens");
        @(negedge clk);advance=0;cancel=0;run(64,333,777);
        launch(96,111,222);repeat(7)@(negedge clk);rst_n=0;
        #1;if(!a_valid)$fatal(1,"input reset became asynchronous");
        @(posedge clk);#1;if(a_valid||bt_valid||read_valid)$fatal(1,"reset left input tokens");
        @(negedge clk);advance=0;rst_n=1;run(32,234,567);
        $display("PASS tb_gemm_input_stream256: continuous reads, tail lanes, same-edge refill, cancel/reset and restart");$finish;
    end
    initial begin #2000000;$fatal(1,"input stream timeout");end
endmodule
