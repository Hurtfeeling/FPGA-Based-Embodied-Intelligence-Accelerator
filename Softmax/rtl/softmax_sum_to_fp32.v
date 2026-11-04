`timescale 1ns/1ps
// Row sum in UQ16.48, bounded to [1,4096]. Five registered stages,
// exact round-to-nearest-even. This is not a general fixed/float converter.
module softmax_sum_to_fp32(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,input wire [63:0] s_sum,
    output wire m_valid,input wire m_ready,output wire [31:0] m_data
);
    reg [4:0] valid_q;
    wire advance=!valid_q[4]||m_ready;
    assign s_ready=rst_n&&!clear&&advance;
    assign m_valid=rst_n&&!clear&&valid_q[4];
    reg [63:0] sum0;
    reg [41:0] sig1;
    reg [3:0] shift1;
    reg [7:0] exp1,exp2,exp3;
    reg [26:0] sig2;
    reg [24:0] rounded3;
    reg [31:0] data4;
    assign m_data=data4;
    function automatic [3:0] lead13(input [12:0] a);
        reg [15:0] x;reg [3:0] n;
        begin
            x={3'b0,a};n=0;
            if(|x[15:8])begin x=x>>8;n=n+4'd8;end
            if(|x[7:4])begin x=x>>4;n=n+4'd4;end
            if(|x[3:2])begin x=x>>2;n=n+4'd2;end
            lead13=n+{3'b0,x[1]};
        end
    endfunction
    function automatic [41:0] shr42(input [41:0] a,input [3:0] n);
        reg [41:0] x;
        begin
            x=a;
            if(n[3])x={8'b0,x[41:9],|x[8:0]};
            if(n[2])x={4'b0,x[41:5],|x[4:0]};
            if(n[1])x={2'b0,x[41:3],|x[2:0]};
            if(n[0])x={1'b0,x[41:2],|x[1:0]};
            shr42=x;
        end
    endfunction
    wire [41:0] shifted=shr42(sig1,shift1);
    always @(posedge clk)begin
        if(!rst_n||clear)valid_q<=0;
        else if(advance)begin
            valid_q<={valid_q[3:0],s_valid};
            sum0<=s_sum;
            sig1<={sum0[63:23],|sum0[22:0]}; // fixed shift 22 with jam
            shift1<=lead13(sum0[60:48]);
            exp1<=8'd127+{4'b0,lead13(sum0[60:48])};
            sig2<=shifted[26:0];exp2<=exp1;
            rounded3<={1'b0,sig2[26:3]}+(sig2[2]&&(sig2[1]||sig2[0]||sig2[3]));
            exp3<=exp2;
            data4<=rounded3[24]?{1'b0,(exp3+8'd1),rounded3[23:1]}:
                {1'b0,exp3,rounded3[22:0]};
        end
    end
endmodule
