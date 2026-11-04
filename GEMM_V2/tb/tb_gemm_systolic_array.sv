`timescale 1ns/1ps
module array_case #(parameter N=4,parameter RESULT_PIPELINE=0) (output reg done);
    localparam RW=(N<=1)?1:$clog2(N),KMAX=37;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,advance=0,in_valid=0;
    wire advance_next=advance,in_valid_next=in_valid;
    wire [N-1:0] a_lane_valid={N{in_valid}},b_lane_valid={N{in_valid}};
    reg [N*8-1:0] a_vec=0,b_vec=0;
    reg [RW-1:0] result_row=0;wire [N*32-1:0] result_data;
    integer aa[0:N-1][0:KMAX-1],bb[0:KMAX-1][0:N-1],expected[0:N-1][0:N-1];
    integer r,c,k,trial,part;reg [N*32-1:0] held;
    gemm_systolic_array #(.N(N),.RESULT_PIPELINE(RESULT_PIPELINE)) dut(.*);
    task step(input bit ce,input bit valid_in);
        begin @(negedge clk);advance=ce;in_valid=valid_in;@(posedge clk);#1;end
    endtask
    task check_all;
        begin for(r=0;r<N;r=r+1)begin
            result_row=r;
            if(RESULT_PIPELINE)begin @(posedge clk);#1;end
            else #0.01;
            for(c=0;c<N;c=c+1)if($signed(result_data[c*32+:32])!==expected[r][c])
                $fatal(1,"N%0d trial%0d row%0d col%0d got%0d expected%0d",N,trial,r,c,$signed(result_data[c*32+:32]),expected[r][c]);
        end end
    endtask
    initial begin
        done=0;repeat(2)@(posedge clk);@(negedge clk);rst_n=1;
        for(trial=0;trial<3;trial=trial+1)begin
            // A new output microtile clears accumulators and pending valid tags.
            @(negedge clk);clear=1;advance=1;in_valid=1;
            @(posedge clk);#1;
            for(r=0;r<N;r=r+1)for(c=0;c<N;c=c+1)expected[r][c]=0;
            check_all();@(negedge clk);clear=0;advance=0;in_valid=0;
            for(k=0;k<KMAX;k=k+1)begin
                for(r=0;r<N;r=r+1)aa[r][k]=(trial==2)?((r+k)%2?127:-128):((r*71+k*23+trial*19)%256)-128;
                for(c=0;c<N;c=c+1)bb[k][c]=(trial==2)?((c+k)%2?-128:127):((k*47+c*37+trial*13)%256)-128;
                for(r=0;r<N;r=r+1)for(c=0;c<N;c=c+1)expected[r][c]=expected[r][c]+aa[r][k]*bb[k][c];
            end
            for(k=0;k<KMAX;k=k+1)begin
                for(r=0;r<N;r=r+1)begin a_vec[r*8+:8]=aa[r][k];b_vec[r*8+:8]=bb[k][r];end
                if(k%3==1)begin
                    // Observe the MACs directly: the independent result mux
                    // may still capture a newly selected row during a pause.
                    for(c=0;c<N;c=c+1)held[c*32+:32]=dut.accum[(k%N)*N+c];
                    step(0,1);
                    for(c=0;c<N;c=c+1)if(dut.accum[(k%N)*N+c]!==held[c*32+:32])
                        $fatal(1,"MAC changed during valid pause");
                    step(0,0);
                    for(c=0;c<N;c=c+1)if(dut.accum[(k%N)*N+c]!==held[c*32+:32])
                        $fatal(1,"MAC changed during invalid pause");
                end
                step(1,1);
                if(k%5==2)begin a_vec='x;b_vec='x;step(1,0);end
            end
            // Unknown bubble payload must never enter a valid accumulation.
            a_vec='x;b_vec='x;
            for(k=0;k<2*N-1;k=k+1)step(1,0);
            @(negedge clk);advance=0;in_valid=0;check_all();step(0,1);check_all();
        end
        result_row=0;#0.01;held=result_data;
        @(negedge clk);rst_n=0;#1;
        if(result_data!==held)$fatal(1,"reset was asynchronous");
        @(posedge clk);#1;
        for(r=0;r<N;r=r+1)for(c=0;c<N;c=c+1)expected[r][c]=0;
        check_all();@(negedge clk);rst_n=1;
        for(r=0;r<N;r=r+1)begin a_vec[r*8+:8]=5;b_vec[r*8+:8]=-7;end
        for(k=0;k<N+1;k=k+1)step(1,1);
        @(negedge clk);advance=0;in_valid=0;
        for(r=0;r<N;r=r+1)begin result_row=r;#0.01;for(c=0;c<N;c=c+1)expected[r][c]=$signed(result_data[c*32+:32]);end
        @(negedge clk);clear=1;advance=1;in_valid=1;
        @(posedge clk);#1;
        for(r=0;r<N;r=r+1)for(c=0;c<N;c=c+1)expected[r][c]=0;
        check_all();
        @(negedge clk);clear=0;advance=0;in_valid=0;
        a_vec='x;b_vec='x;
        for(k=0;k<2*N+2;k=k+1)step(1,0);
        @(negedge clk);advance=0;check_all();
        $display("PASS single-K array N%0d pipeline%0d: signed/extreme MAC, unknown invalid payload, pause, synchronous reset and valid-only flush",N,RESULT_PIPELINE);done=1;
    end
endmodule
// Maximum legal K checks both INT8 product extrema beyond narrow accumulators.
module pe_range_case(output reg done);
    localparam KMAX=16352;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,advance=0,in_valid=0;
    reg signed [7:0] a=0,b=0;
    wire [31:0] result;
    integer trial,k,expected;
    reg [31:0] held;
    gemm_pe_mac dut(.*);
    task step(input bit ce,input bit valid_in,input integer av,input integer bv);
        begin
            @(negedge clk);advance=ce;in_valid=valid_in;a=av;b=bv;
            @(posedge clk);#1;
        end
    endtask
    initial begin
        done=0;repeat(2)@(posedge clk);@(negedge clk);rst_n=1;
        for(trial=0;trial<2;trial=trial+1)begin
            @(negedge clk);clear=1;advance=1;in_valid=1;
            @(posedge clk);#1;if(result!==32'd0)$fatal(1,"PE clear failed");
            @(negedge clk);clear=0;advance=0;in_valid=0;
            for(k=0;k<KMAX;k=k+1)begin
                if(k%13==0)begin
                    held=result;step(0,1,127,127);
                    if(result!==held)$fatal(1,"PE range paused accumulator changed");
                end
                step(1,1,-128,(trial==0)?-128:127);
                if(k%17==0)step(1,0,127,-128);
            end
            step(1,0,0,0);
            // Independent constants: 16352*(-128)*(-128 or 127).
            expected=(trial==0)?267911168:-265818112;
            if($signed(result)!==expected)
                $fatal(1,"PE max K trial%0d got%0d expected%0d",trial,$signed(result),expected);
        end
        $display("PASS PE INT32 max K=16352: positive/negative extrema, bubbles and stalls");done=1;
    end
endmodule
module tb_gemm_systolic_array;
    wire d4,d32,dpipe,drange;pe_range_case t_range(drange);array_case #(.N(4)) t4(d4);array_case #(.N(32)) t32(d32);
    array_case #(.N(32),.RESULT_PIPELINE(1)) tp(dpipe);
    initial begin wait(d4&&d32&&dpipe&&drange);$display("PASS tb_gemm_systolic_array");$finish;end
    initial begin #1000000;$fatal(1,"array timeout");end
endmodule
