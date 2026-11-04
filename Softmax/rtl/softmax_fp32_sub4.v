`timescale 1ns/1ps
// Four-lane binary32 x-max subtractor. Each lane has ten elastic registers:
// capture, decode, magnitude select, align, add/sub, leading-one detection,
// normalization, gradual-underflow shift, RNE, and packing. II=1.
// Subnormal inputs are DAZ; subnormal results are rounded before FTZ.
module softmax_fp32_sub4(
    input wire clk,input wire rst_n,input wire clear,
    input wire s_valid,output wire s_ready,
    input wire [127:0] s_data,input wire [31:0] s_max,
    input wire [3:0] s_keep,input wire [9:0] s_tag,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_data,output wire [3:0] m_keep,
    output wire [9:0] m_tag
);
    // A bounded barrel shifter with sticky accumulation. Each step has a
    // fixed wiring shift and reduction tree; no 32-bit integer comparisons.
    function automatic [26:0] shr27(input [26:0] a,input [7:0] sh);
        reg [26:0] x;
        begin
            x=a;
            if(sh[4]) x={16'b0,x[26:17],|x[16:0]};
            if(sh[3]) x={8'b0,x[26:9],|x[8:0]};
            if(sh[2]) x={4'b0,x[26:5],|x[4:0]};
            if(sh[1]) x={2'b0,x[26:3],|x[2:0]};
            if(sh[0]) x={1'b0,x[26:2],|x[1:0]};
            shr27=(|sh[7:5])?{26'b0,|a}:x;
        end
    endfunction
    function automatic [4:0] msb28(input [27:0] a);
        reg [31:0] x; reg [4:0] pos;
        begin
            x={4'b0,a}; pos=0;
            if(|x[31:16]) begin x=x>>16;pos[4]=1;end
            if(|x[15:8]) begin x=x>>8;pos[3]=1;end
            if(|x[7:4]) begin x=x>>4;pos[2]=1;end
            if(|x[3:2]) begin x=x>>2;pos[1]=1;end
            if(x[1]) pos[0]=1;
            msb28=pos;
        end
    endfunction

    reg [9:0] valid_q;
    wire [9:0] ready_q;
    assign ready_q[9]=!valid_q[9]||m_ready;
    genvar st;
    generate for(st=0;st<9;st=st+1) begin:g_ready
        assign ready_q[st]=!valid_q[st]||ready_q[st+1];
    end endgenerate
    assign s_ready=rst_n&&!clear&&ready_q[0];
    assign m_valid=rst_n&&!clear&&valid_q[9];
    reg [3:0] keep_q[0:9]; reg [9:0] tag_q[0:9];
    reg [127:0] d0; reg [31:0] max0[0:3];
    reg [7:0] ea1[0:3],eb1[0:3]; reg [23:0] ma1[0:3],mb1[0:3];
    reg sa1[0:3],sb1[0:3];
    reg [7:0] exp2[0:3],shift2[0:3]; reg [26:0] hi2[0:3],lo2[0:3];
    reg sign2[0:3],add2[0:3];
    reg [26:0] hi3[0:3],lo3[0:3]; reg [7:0] exp3[0:3]; reg sign3[0:3],add3[0:3];
    reg [27:0] raw4[0:3]; reg [7:0] exp4[0:3]; reg sign4[0:3];
    reg [27:0] raw5[0:3]; reg [7:0] exp5[0:3]; reg sign5[0:3]; reg [4:0] lead5[0:3];
    reg [26:0] norm6[0:3]; reg signed [9:0] exp6[0:3]; reg sign6[0:3];
    reg [26:0] norm7[0:3]; reg [8:0] exp7[0:3]; reg sign7[0:3];
    reg [24:0] rounded8[0:3]; reg [8:0] exp8[0:3]; reg sign8[0:3];
    reg [127:0] data9;
    assign m_data=data9; assign m_keep=keep_q[9]; assign m_tag=tag_q[9];

    integer i,s; reg [31:0] a,b; reg high_a; reg [4:0] left_shift;
    reg signed [9:0] denorm_shift; reg [7:0] denorm_shift8;
    reg [8:0] packed_exp; reg [23:0] packed_mant;
    always @(posedge clk) begin
        if(!rst_n||clear) valid_q<=0;
        else begin
            if(ready_q[0]) begin
                valid_q[0]<=s_valid;
                if(s_valid) begin
                    d0<=s_data;keep_q[0]<=s_keep;tag_q[0]<=s_tag;
                    for(i=0;i<4;i=i+1)max0[i]<=s_max;
                end
            end
            for(s=1;s<10;s=s+1) if(ready_q[s]) begin
                valid_q[s]<=valid_q[s-1];
                if(valid_q[s-1]) begin keep_q[s]<=keep_q[s-1];tag_q[s]<=tag_q[s-1];end
            end
            if(ready_q[1]&&valid_q[0]) begin
                for(i=0;i<4;i=i+1) begin
                    a=d0[i*32+:32];b=max0[i];
                    ea1[i]<=a[30:23];eb1[i]<=b[30:23];
                    ma1[i]<=a[30:23]==0?24'b0:{1'b1,a[22:0]};
                    mb1[i]<=b[30:23]==0?24'b0:{1'b1,b[22:0]};
                    sa1[i]<=a[31];sb1[i]<=b[31];
                    // Core rejects NaN/+Inf. -Inf is an attention mask and
                    // must emerge with keep=0 so exp produces exactly zero.
                    keep_q[1][i]<=keep_q[0][i]&&(a[30:23]!=8'hff)&&(b[30:23]!=8'hff);
                end
            end
            if(ready_q[2]&&valid_q[1]) begin
                for(i=0;i<4;i=i+1) begin
                    high_a=(ea1[i]>eb1[i])||((ea1[i]==eb1[i])&&(ma1[i]>=mb1[i]));
                    exp2[i]<=high_a?ea1[i]:eb1[i];
                    shift2[i]<=high_a?(ea1[i]-eb1[i]):(eb1[i]-ea1[i]);
                    hi2[i]<=high_a?{ma1[i],3'b0}:{mb1[i],3'b0};
                    lo2[i]<=high_a?{mb1[i],3'b0}:{ma1[i],3'b0};
                    add2[i]<=sa1[i]!=sb1[i];
                    if(ea1[i]==0&&eb1[i]==0)sign2[i]<=sa1[i]&~sb1[i];
                    else sign2[i]<=(sa1[i]!=sb1[i])?sa1[i]:(high_a?sa1[i]:~sa1[i]);
                end
            end
            if(ready_q[3]&&valid_q[2]) begin
                for(i=0;i<4;i=i+1) begin
                    hi3[i]<=hi2[i];lo3[i]<=shr27(lo2[i],shift2[i]);
                    exp3[i]<=exp2[i];sign3[i]<=sign2[i];add3[i]<=add2[i];
                end
            end
            if(ready_q[4]&&valid_q[3]) begin
                for(i=0;i<4;i=i+1) begin
                    raw4[i]<=add3[i]?({1'b0,hi3[i]}+{1'b0,lo3[i]}):({1'b0,hi3[i]}-{1'b0,lo3[i]});
                    exp4[i]<=exp3[i];sign4[i]<=sign3[i];
                end
            end
            if(ready_q[5]&&valid_q[4]) begin
                for(i=0;i<4;i=i+1) begin
                    raw5[i]<=raw4[i];exp5[i]<=exp4[i];sign5[i]<=sign4[i];lead5[i]<=msb28(raw4[i]);
                end
            end
            if(ready_q[6]&&valid_q[5]) begin
                for(i=0;i<4;i=i+1) begin
                    sign6[i]<=sign5[i];
                    if(raw5[i]==0) begin
                        norm6[i]<=0;exp6[i]<=0;
                        // Exact cancellation of nonzero finite values is +0.
                        if(exp5[i]!=0)sign6[i]<=0;
                    end else if(raw5[i][27]) begin
                        norm6[i]<={raw5[i][27:2],|raw5[i][1:0]};
                        exp6[i]<=$signed({2'b0,exp5[i]})+10'sd1;
                    end else begin
                        left_shift=5'd26-lead5[i];
                        norm6[i]<=raw5[i][26:0]<<left_shift;
                        exp6[i]<=$signed({2'b0,exp5[i]})-$signed({5'b0,left_shift});
                    end
                end
            end
            if(ready_q[7]&&valid_q[6]) begin
                for(i=0;i<4;i=i+1) begin
                    sign7[i]<=sign6[i];
                    if(exp6[i]<=0) begin
                        denorm_shift=10'sd1-exp6[i];
                        denorm_shift8=(denorm_shift > 10'sd31) ? 8'd32 : denorm_shift[7:0];
                        norm7[i]<=shr27(norm6[i],denorm_shift8);exp7[i]<=9'd1;
                    end else begin norm7[i]<=norm6[i];exp7[i]<=exp6[i][8:0];end
                end
            end
            if(ready_q[8]&&valid_q[7]) begin
                for(i=0;i<4;i=i+1) begin
                    rounded8[i]<={1'b0,norm7[i][26:3]}+
                        (norm7[i][2]&&(norm7[i][1]||norm7[i][0]||norm7[i][3]));
                    exp8[i]<=exp7[i];sign8[i]<=sign7[i];
                end
            end
            if(ready_q[9]&&valid_q[8]) begin
                for(i=0;i<4;i=i+1) begin
                    packed_exp=exp8[i]+rounded8[i][24];
                    packed_mant=rounded8[i][24]?rounded8[i][24:1]:rounded8[i][23:0];
                    if(!keep_q[8][i])data9[i*32+:32]<=0;
                    else if(packed_exp>=9'd255)data9[i*32+:32]<={sign8[i],8'hff,23'b0};
                    else if(!packed_mant[23])data9[i*32+:32]<={sign8[i],31'b0};
                    else data9[i*32+:32]<={sign8[i],packed_exp[7:0],packed_mant[22:0]};
                end
            end
        end
    end
endmodule

