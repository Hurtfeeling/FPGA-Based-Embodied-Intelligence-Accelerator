`timescale 1ns/1ps
// Four-lane binary32 multiply by a common binary32 scale. Eight registered
// stages, II=1: capture, decode, DSP partial products, product combine,
// normalize, underflow preparation, RNE, pack. DAZ/FTZ semantics match
// fp32_mul() in softmax_fp32_funcs.vh, including signed zero and special values.
// Splitting 24x24 into 24x17 and 24x7 gives each DSP a registered product;
// the 48-bit partial-product addition is in its own following stage.
module softmax_fp32_mul4(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,
    input wire [127:0] s_data,input wire [31:0] s_scale,
    input wire [3:0] s_keep,input wire [9:0] s_tag,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_data,output wire [3:0] m_keep,
    output wire [9:0] m_tag
);
    localparam [1:0] NORMAL=2'd0,ZERO=2'd1,INFINITY=2'd2,NAN=2'd3;
    reg [7:0] valid_q;
    reg [3:0] keep_q[0:7];
    reg [9:0] tag_q[0:7];
    reg [127:0] data0,data7;
    reg [31:0] scale0;
    reg [23:0] mant_a1[0:3],mant_b1[0:3];
    (* use_dsp="yes" *) reg [40:0] product_lo2[0:3];
    (* use_dsp="yes" *) reg [30:0] product_hi2[0:3];
    reg [47:0] product3[0:3];
    reg [26:0] sig4[0:3],sig5[0:3];
    reg [24:0] rounded6[0:3];
    reg signed [9:0] exponent1[0:3],exponent2[0:3],exponent3[0:3];
    reg signed [9:0] exponent4[0:3],exponent5[0:3],exponent6[0:3];
    reg [3:0] sign1,sign2,sign3,sign4,sign5,sign6;
    reg [1:0] special1[0:3],special2[0:3],special3[0:3];
    reg [1:0] special4[0:3],special5[0:3],special6[0:3];
    wire advance=!valid_q[7]||m_ready;
    assign s_ready=rst_n&&!clear&&advance;
    assign m_valid=rst_n&&!clear&&valid_q[7];
    assign m_data=data7;
    assign m_keep=keep_q[7];
    assign m_tag=tag_q[7];
    integer stage,lane;
    reg [31:0] a,b;
    reg a_zero,b_zero,a_inf,b_inf,a_nan,b_nan;
    reg [9:0] final_exp;
    reg [23:0] final_mant;
    always @(posedge clk) begin
        if(!rst_n||clear) valid_q<=0;
        else if(advance) begin
            valid_q<={valid_q[6:0],s_valid};
            keep_q[0]<=s_keep;tag_q[0]<=s_tag;
            for(stage=1;stage<8;stage=stage+1) begin
                keep_q[stage]<=keep_q[stage-1];tag_q[stage]<=tag_q[stage-1];
            end
            data0<=s_data;scale0<=s_scale;
            for(lane=0;lane<4;lane=lane+1) begin
                // S1: classification and decode. Exponent-zero input is DAZ.
                a=data0[lane*32+:32];b=scale0;
                a_zero=a[30:23]==0;b_zero=b[30:23]==0;
                a_inf=(&a[30:23])&&!(|a[22:0]);
                b_inf=(&b[30:23])&&!(|b[22:0]);
                a_nan=(&a[30:23])&&(|a[22:0]);
                b_nan=(&b[30:23])&&(|b[22:0]);
                mant_a1[lane]<={1'b1,a[22:0]};
                mant_b1[lane]<={1'b1,b[22:0]};
                exponent1[lane]<=$signed({2'b0,a[30:23]})+
                                 $signed({2'b0,b[30:23]})-10'sd127;
                sign1[lane]<=a[31]^b[31];
                if(a_nan||b_nan||(a_inf&&b_zero)||(b_inf&&a_zero))
                    special1[lane]<=NAN;
                else if(a_inf||b_inf) special1[lane]<=INFINITY;
                else if(a_zero||b_zero) special1[lane]<=ZERO;
                else special1[lane]<=NORMAL;

                // S2: independent registered DSP products (no DSP cascade).
                product_lo2[lane]<=mant_a1[lane]*mant_b1[lane][16:0];
                product_hi2[lane]<=mant_a1[lane]*mant_b1[lane][23:17];
                exponent2[lane]<=exponent1[lane];
                sign2[lane]<=sign1[lane];special2[lane]<=special1[lane];

                // S3: full 48-bit product, independently registered adder.
                product3[lane]<={7'b0,product_lo2[lane]}+
                                {product_hi2[lane],17'b0};
                exponent3[lane]<=exponent2[lane];
                sign3[lane]<=sign2[lane];special3[lane]<=special2[lane];

                // S4: a normalized product needs at most one-bit shifting.
                if(product3[lane][47]) begin
                    sig4[lane]<={product3[lane][47:22],|product3[lane][21:0]};
                    exponent4[lane]<=exponent3[lane]+10'sd1;
                end else begin
                    sig4[lane]<={product3[lane][46:21],|product3[lane][20:0]};
                    exponent4[lane]<=exponent3[lane];
                end
                sign4[lane]<=sign3[lane];special4[lane]<=special3[lane];

                // S5: FTZ means only exponent==0 can round back up to the
                // minimum normal value. Smaller results always flush to zero.
                // This replaces a general subnormal barrel shift with one
                // fixed shift, preserving the minimum-normal RNE boundary.
                if(exponent4[lane]<0) begin
                    sig5[lane]<=0;exponent5[lane]<=10'sd1;
                end else if(exponent4[lane]==0) begin
                    sig5[lane]<={1'b0,sig4[lane][26:2],|sig4[lane][1:0]};
                    exponent5[lane]<=10'sd1;
                end else begin
                    sig5[lane]<=sig4[lane];exponent5[lane]<=exponent4[lane];
                end
                sign5[lane]<=sign4[lane];special5[lane]<=special4[lane];

                // S6: round-to-nearest-even with guard/round/sticky bits.
                rounded6[lane]<={1'b0,sig5[lane][26:3]}+
                    (sig5[lane][2]&&(sig5[lane][1]||sig5[lane][0]||sig5[lane][3]));
                exponent6[lane]<=exponent5[lane];
                sign6[lane]<=sign5[lane];special6[lane]<=special5[lane];

                // S7: carry correction, overflow/FTZ and binary32 pack.
                final_exp=exponent6[lane]+rounded6[lane][24];
                final_mant=rounded6[lane][24]?rounded6[lane][24:1]:rounded6[lane][23:0];
                if(!keep_q[6][lane]) data7[lane*32+:32]<=0;
                else if(special6[lane]==NAN) data7[lane*32+:32]<=32'h7fc00000;
                else if(special6[lane]==INFINITY||
                        (special6[lane]==NORMAL&&final_exp>=10'd255))
                    data7[lane*32+:32]<={sign6[lane],8'hff,23'b0};
                else if(special6[lane]==ZERO||!final_mant[23])
                    data7[lane*32+:32]<={sign6[lane],31'b0};
                else data7[lane*32+:32]<={sign6[lane],final_exp[7:0],final_mant[22:0]};
            end
        end
    end
endmodule
