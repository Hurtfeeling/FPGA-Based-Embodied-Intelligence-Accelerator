`timescale 1ns/1ps
module tb_softmax_sum4_random;
    `include "count.vh"
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,s_valid=0,s_last=0,m_ready=0;
    reg [127:0] s_data=0;
    wire s_ready,m_valid;wire [63:0] m_sum;
    softmax_sum4 dut(.clk(clk),.rst_n(rst_n),.clear(1'b0),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_last(s_last),
        .m_valid(m_valid),.m_ready(m_ready),.m_sum(m_sum));
    reg [127:0] data[0:NBEATS-1];
    reg [63:0] sums[0:63];reg [10:0] lengths[0:63];
    integer row,beat,index=0,j;reg [63:0] held;
    initial begin
        $readmemh("Softmax/tb/build/sum4_random/data.mem",data);
        $readmemh("Softmax/tb/build/sum4_random/sums.mem",sums);
        $readmemh("Softmax/tb/build/sum4_random/lengths.mem",lengths);
        repeat(3)@(negedge clk);rst_n=1;
        for(row=0;row<64;row=row+1)begin
            for(beat=0;beat<lengths[row];beat=beat+1)begin
                if(beat%17==0)begin s_valid=0;repeat(2)@(negedge clk);end
                s_valid=1;s_data=data[index];s_last=beat==lengths[row]-1;index=index+1;
                @(posedge clk);while(!s_ready)@(posedge clk);
                @(negedge clk);
            end
            s_valid=0;s_last=0;wait(m_valid);@(negedge clk);held=m_sum;
            if(m_sum!==sums[row])
                $fatal(1,"sum4 random row=%0d got=%h expected=%h",row,m_sum,sums[row]);
            for(j=0;j<5;j=j+1)begin
                @(negedge clk);
                if(!m_valid||m_sum!==held||s_ready)$fatal(1,"sum4 changed under stall");
            end
            m_ready=1;@(negedge clk);m_ready=0;
        end
        $display("PASS sum4 random: 64 rows, %0d beats, independent integer RNE, bubbles/stalls",NBEATS);$finish;
    end
    initial begin #2000000;$fatal(1,"sum4 random timeout");end
endmodule
