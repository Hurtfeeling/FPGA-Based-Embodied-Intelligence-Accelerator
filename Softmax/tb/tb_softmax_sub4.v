`timescale 1ns/1ps
module tb_softmax_sub4;
    `include "softmax_fp32_funcs.vh"
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,s_valid=0,m_ready=0;
    reg [127:0] s_data=0;reg [31:0] s_max=0;reg [3:0] s_keep=0;reg [9:0] s_tag=0;
    wire s_ready,m_valid;wire [127:0] m_data;wire [3:0] m_keep;wire [9:0] m_tag;
    softmax_fp32_sub4 dut(.clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_max(s_max),.s_keep(s_keep),.s_tag(s_tag),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data),.m_keep(m_keep),.m_tag(m_tag));
    reg [295:0] vectors[0:8191];
    integer phase=0,index,received=0,lane,run=0,max_run=0,stall_cycles=0;
    reg stalled=0;reg [127:0] held_data;reg [3:0] held_keep;reg [9:0] held_tag;
    always @(negedge clk) if(phase==1) m_ready=($random&7)!=0;
    always @(posedge clk) begin
        if(!rst_n||clear) begin stalled<=0;run=0;end
        else begin
            if(stalled&&(!m_valid||m_data!==held_data||m_keep!==held_keep||m_tag!==held_tag))
                $fatal(1,"sub4 output changed under backpressure");
            stalled<=m_valid&&!m_ready;
            held_data<=m_data;held_keep<=m_keep;held_tag<=m_tag;
            if(phase==1) begin
                if(s_valid&&s_ready) begin run=run+1;if(run>max_run)max_run=run;end else run=0;
                if(m_valid&&!m_ready)stall_cycles=stall_cycles+1;
                if(m_valid&&m_ready) begin
                    if(m_tag!==received[9:0]||m_keep!==vectors[received][295:292]||m_data!==vectors[received][291:164])
                        $fatal(1,"sub4 vector=%0d tag=%h keep=%h/%h got=%h expected=%h x=%h max=%h",received,m_tag,m_keep,vectors[received][295:292],m_data,vectors[received][291:164],vectors[received][127:0],vectors[received][159:128]);
                    received=received+1;
                end
            end
        end
    end
    initial begin
        $readmemh("Softmax/tb/build/sub4/vectors.mem",vectors);
        // Independent Python values must also agree with the established
        // helper; DUT comparison below uses those exact expected bits.
        for(index=0;index<8192;index=index+1)for(lane=0;lane<4;lane=lane+1)
            if(vectors[index][292+lane]&&fp32_sub(vectors[index][32*lane+:32],vectors[index][159:128])!==vectors[index][164+32*lane+:32])
                $fatal(1,"reference mismatch vector=%0d lane=%0d",index,lane);
        repeat(3)@(negedge clk);rst_n=1;
        s_valid=1;s_data={4{32'h3f800000}};s_max=32'h40000000;s_keep=15;
        repeat(24)@(negedge clk);
        if(s_ready)$fatal(1,"full stalled sub4 pipeline did not backpressure");
        clear=1;s_valid=0;@(negedge clk);clear=0;m_ready=1;
        repeat(12)@(negedge clk);
        if(m_valid)$fatal(1,"clear leaked stale token");
        phase=1;
        for(index=0;index<8192;index=index+1)begin
            if(index%31==0)begin s_valid=0;repeat(3)@(negedge clk);end
            s_valid=1;s_data=vectors[index][127:0];s_max=vectors[index][159:128];
            s_keep=vectors[index][163:160];s_tag=index[9:0];
            @(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);
        end
        s_valid=0;wait(received==8192);@(negedge clk);phase=2;
        if(max_run<16||stall_cycles<100)$fatal(1,"insufficient continuous/stalled traffic run=%0d stalls=%0d",max_run,stall_cycles);
        $display("PASS sub4 pipeline: %0d vectors, II=1 burst=%0d, stalled cycles=%0d, clear cancellation",received,max_run,stall_cycles);$finish;
    end
    initial begin #3000000;$fatal(1,"sub4 timeout");end
endmodule
