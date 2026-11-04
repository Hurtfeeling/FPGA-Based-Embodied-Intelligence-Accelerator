`timescale 1ns/1ps
// Four-lane exp(x-max), II=1. Dedicated FP pipelines isolate subtraction
// and log2(e) multiplication. The LUT backend has nine short stages:
// shift decode, fixed conversion, ROM, ROM output register, interpolation
// multiply, drop rounding, interpolation subtract, significand RNE, pack.
module softmax_exp4 #(
    // Add exp_lut.mem as a Vivado Memory Initialization File, or override
    // this parameter with an absolute path for non-project builds.
    parameter LUT_FILE="exp_lut.mem"
)(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,
    input wire [127:0] s_data,input wire [31:0] s_max,
    input wire [3:0] s_keep,input wire [9:0] s_tag,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_data,output wire [9:0] m_tag
);
    (* rom_style="block" *) reg [63:0] lut01[0:1023];
    (* rom_style="block" *) reg [63:0] lut23[0:1023];
    initial begin
        $readmemh(LUT_FILE,lut01);
        $readmemh(LUT_FILE,lut23);
    end
    wire sub_valid,sub_ready,mul_s_ready,mul_valid;
    wire advance;
    wire [127:0] sub_data,abs_data,mul_data;
    wire [3:0] sub_keep,mul_keep;
    wire [9:0] sub_tag,mul_tag;
    softmax_fp32_sub4 u_sub(
        .clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(sub_ready),.s_data(s_data),.s_max(s_max),
        .s_keep(s_keep),.s_tag(s_tag),.m_valid(sub_valid),.m_ready(mul_s_ready),
        .m_data(sub_data),.m_keep(sub_keep),.m_tag(sub_tag));
    genvar g;
    generate for(g=0;g<4;g=g+1) begin:g_abs
        assign abs_data[g*32+:32]={1'b0,sub_data[g*32+:31]};
    end endgenerate
    softmax_fp32_mul4 u_log2e(
        .clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(sub_valid),.s_ready(mul_s_ready),.s_data(abs_data),
        .s_scale(32'h3fb8aa3b),.s_keep(sub_keep),.s_tag(sub_tag),
        .m_valid(mul_valid),.m_ready(advance),.m_data(mul_data),
        .m_keep(mul_keep),.m_tag(mul_tag));

    reg [8:0] valid_q;
    assign advance=!valid_q[8]||m_ready;
    reg [9:0] tags[0:8];
    reg [3:0] zero_q[0:8];
    reg [23:0] mant0[0:3];
    reg [7:0] shift0[0:3]; reg [3:0] right0;
    reg [31:0] fixed1[0:3];
    reg [63:0] pair2[0:3],pair3[0:3];
    reg [12:0] frac2[0:3],frac3[0:3];
    reg [6:0] n2[0:3],n3[0:3],n4[0:3],n5[0:3],n6[0:3];
    reg [30:0] base4[0:3],base5[0:3],q6[0:3];
    (* use_dsp="yes" *) reg [32:0] product4[0:3];
    reg [20:0] drop5[0:3];
    reg [24:0] rounded7[0:3];
    reg signed [9:0] exponent7[0:3];
    reg [127:0] data8;
    assign s_ready=rst_n&&!clear&&sub_ready;
    assign m_valid=rst_n&&!clear&&valid_q[8];
    assign m_data=data8;assign m_tag=tags[8];

    integer lane,i;
    reg [31:0] t;
    reg signed [9:0] e;
    reg [23:0] rounded;
    always @(posedge clk) begin
        if(!rst_n||clear) begin
            valid_q<=0;
            for(i=0;i<9;i=i+1) zero_q[i]<=0;
        end else if(advance) begin
            valid_q<={valid_q[7:0],mul_valid};
            tags[0]<=mul_tag;
            for(i=1;i<9;i=i+1) begin tags[i]<=tags[i-1];zero_q[i]<=zero_q[i-1];end
            for(lane=0;lane<4;lane=lane+1) begin
                t=mul_data[lane*32+:32];
                mant0[lane]<={1'b1,t[22:0]};
                right0[lane]<=t[30:23]<8'd127;
                shift0[lane]<=(t[30:23]<8'd127)?8'd127-t[30:23]:t[30:23]-8'd127;
                zero_q[0][lane]<=!mul_keep[lane]||(t[30:0]>31'h42fc0000);
                // Zero/DAZ gives a shift of 127, which also maps to zero.
                fixed1[lane]<=right0[lane]?({8'b0,mant0[lane]}>>shift0[lane]):
                    ({8'b0,mant0[lane]}<<shift0[lane]);
                n2[lane]<=fixed1[lane][29:23];frac2[lane]<=fixed1[lane][12:0];
                pair3[lane]<=pair2[lane];frac3[lane]<=frac2[lane];n3[lane]<=n2[lane];
                // Every generated LUT delta fits 20 bits (max 726572).
                // 20x13 fits one DSP48E2 instead of a 32x13 DSP cascade.
                product4[lane]<=pair3[lane][19:0]*frac3[lane];
                base4[lane]<=pair3[lane][62:32];n4[lane]<=n3[lane];
                drop5[lane]<=product4[lane][32:13]+
                    (product4[lane][12]&&((|product4[lane][11:0])||product4[lane][13]));
                base5[lane]<=base4[lane];n5[lane]<=n4[lane];
                q6[lane]<=base5[lane]-drop5[lane];n6[lane]<=n5[lane];
                // Interpolation is in [0.5,1], so normalization is a 2:1
                // select, not a general leading-one detector/barrel shifter.
                if(q6[lane][30]) begin
                    rounded7[lane]<={1'b0,q6[lane][30:7]}+
                        (q6[lane][6]&&((|q6[lane][5:0])||q6[lane][7]));
                    exponent7[lane]<=10'sd127-$signed({3'b0,n6[lane]});
                end else begin
                    rounded7[lane]<={1'b0,q6[lane][29:6]}+
                        (q6[lane][5]&&((|q6[lane][4:0])||q6[lane][6]));
                    exponent7[lane]<=10'sd126-$signed({3'b0,n6[lane]});
                end
                e=exponent7[lane]+$signed({9'b0,rounded7[lane][24]});
                rounded=rounded7[lane][24]?rounded7[lane][24:1]:rounded7[lane][23:0];
                if(zero_q[7][lane]||e<=0) data8[lane*32+:32]<=0;
                else data8[lane*32+:32]<={1'b0,e[7:0],rounded[22:0]};
            end
            pair2[0]<=lut01[fixed1[0][22:13]];
            pair2[1]<=lut01[fixed1[1][22:13]];
            pair2[2]<=lut23[fixed1[2][22:13]];
            pair2[3]<=lut23[fixed1[3][22:13]];
        end
    end
endmodule
