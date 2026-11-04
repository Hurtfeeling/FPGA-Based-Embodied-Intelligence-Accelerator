`timescale 1ns/1ps
module tb_softmax_recip;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0,s_valid=0,m_ready=0;
    reg [31:0] s_data=0;
    wire s_ready,m_valid;wire [31:0] m_data;
    softmax_recip dut(.clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),
        .m_valid(m_valid),.m_ready(m_ready),.m_data(m_data));
    reg [63:0] vectors[0:4095];integer i,j;reg [31:0] held;
    initial begin
        $readmemh("Softmax/tb/build/recip/vectors.mem",vectors);
        repeat(3)@(negedge clk);rst_n=1;
        // Cancel a divider in flight, then require idle and no stale result.
        s_valid=1;s_data=32'h43f00100;@(negedge clk);s_valid=0;
        repeat(20)@(negedge clk);clear=1;@(negedge clk);clear=0;
        repeat(60)@(negedge clk);
        if(m_valid||!s_ready)$fatal(1,"reciprocal clear failed");
        for(i=0;i<4096;i=i+1)begin
            s_valid=1;s_data=vectors[i][63:32];
            @(posedge clk);while(!s_ready)@(posedge clk);
            @(negedge clk);s_valid=0;
            wait(m_valid);@(negedge clk);held=m_data;
            for(j=0;j<3;j=j+1)begin
                @(negedge clk);
                if(!m_valid||m_data!==held)$fatal(1,"reciprocal changed under stall");
            end
            if(m_data!==vectors[i][31:0])
                $fatal(1,"reciprocal[%0d] input=%h got=%h expected=%h",i,s_data,m_data,vectors[i][31:0]);
            m_ready=1;@(negedge clk);m_ready=0;
        end
        $display("PASS reciprocal: 4096 exact cases, all exponents/signs, DAZ/FTZ/specials, stall and clear");$finish;
    end
    initial begin #3000000;$fatal(1,"reciprocal timeout");end
endmodule
