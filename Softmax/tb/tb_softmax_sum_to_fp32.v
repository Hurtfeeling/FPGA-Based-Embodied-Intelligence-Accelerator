`timescale 1ns/1ps
module tb_softmax_sum_to_fp32;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,s_valid=0,m_ready=0;
    reg [63:0] s_sum=0;
    wire s_ready,m_valid;wire [31:0] m_data;
    softmax_sum_to_fp32 dut(.clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_sum(s_sum),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data));
    reg [95:0] vectors[0:8191];
    integer i,received=0,phase=0,stalls=0;
    reg stalled=0;reg [31:0] held;
    always @(negedge clk) if(phase==1)m_ready=($random&3)!=0;
    always @(posedge clk) begin
        if(!rst_n||clear)stalled<=0;
        else begin
            if(stalled&&(!m_valid||m_data!==held))$fatal(1,"sum_to_fp32 changed under stall");
            stalled<=m_valid&&!m_ready;held<=m_data;
            if(phase==1&&m_valid&&!m_ready)stalls=stalls+1;
            if(phase==1&&m_valid&&m_ready)begin
                if(m_data!==vectors[received][31:0])
                    $fatal(1,"sum_to_fp32[%0d] input=%h got=%h expected=%h",received,vectors[received][95:32],m_data,vectors[received][31:0]);
                received=received+1;
            end
        end
    end
    initial begin
        $readmemh("Softmax/tb/build/sum_to_fp32/vectors.mem",vectors);
        repeat(3)@(negedge clk);rst_n=1;
        // Fill the entire blocked pipeline and cancel every queued token.
        s_valid=1;s_sum=64'h0001000000000000;
        repeat(12)@(negedge clk);
        if(s_ready)$fatal(1,"sum_to_fp32 full pipe did not backpressure");
        clear=1;s_valid=0;@(negedge clk);clear=0;m_ready=1;
        repeat(8)@(negedge clk);
        if(m_valid)$fatal(1,"sum_to_fp32 clear leaked a token");
        phase=1;
        for(i=0;i<8192;i=i+1)begin
            if(i%17==0)begin s_valid=0;repeat(2)@(negedge clk);end
            s_valid=1;s_sum=vectors[i][95:32];
            @(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);
        end
        s_valid=0;wait(received==8192);@(negedge clk);phase=2;
        if(stalls<100)$fatal(1,"sum_to_fp32 insufficient backpressure coverage");
        $display("PASS sum_to_fp32: 8192 exact RNE vectors, stalls=%0d, bubbles and clear",stalls);$finish;
    end
    initial begin #2000000;$fatal(1,"sum_to_fp32 timeout");end
endmodule
