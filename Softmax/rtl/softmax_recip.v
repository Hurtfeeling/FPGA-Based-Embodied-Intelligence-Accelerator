`timescale 1ns/1ps
// One reciprocal per row: 48 restoring divide steps followed by separate
// remainder comparison, RNE increment, and binary32 packing cycles. Positive
// Softmax row sums are in [1,4096]. DAZ/FTZ and IEEE special values are defined
// for standalone use as well. No combinational division operator is used.
module softmax_recip(
    input wire clk,rst_n,clear,
    input wire s_valid,output wire s_ready,input wire [31:0] s_data,
    output reg m_valid,input wire m_ready,output reg [31:0] m_data
);
    localparam IDLE=3'd0,DIVIDE=3'd1,COMPARE=3'd2,ROUND=3'd3,PACK=3'd4;
    reg [2:0] state;
    reg sign_q;
    reg [47:0] dividend;
    reg [23:0] divisor,quotient;
    reg [24:0] remainder;
    reg [5:0] steps;
    reg signed [9:0] exponent_q;
    reg round_up;
    reg [24:0] rounded_q;
    wire [24:0] trial={remainder[23:0],dividend[47]};
    // Borrow simultaneously supplies the comparison and subtract result.
    // This removes a comparator carry chain in series with a subtractor.
    wire [25:0] trial_difference={1'b0,trial}-{2'b0,divisor};
    wire [24:0] remainder_next=trial_difference[25] ? trial : trial_difference[24:0];
    wire [23:0] quotient_next={quotient[22:0],!trial_difference[25]};
    wire [25:0] twice_remainder={remainder,1'b0};
    wire [31:0] input_value=(s_data[30:23]==0) ? {s_data[31],31'b0} : s_data;
    wire [8:0] power_exponent=9'd254-{1'b0,input_value[30:23]};
    wire signed [9:0] packed_exponent=exponent_q+$signed({9'b0,rounded_q[24]});
    wire [23:0] packed_mantissa=rounded_q[24] ? rounded_q[24:1] : rounded_q[23:0];
    assign s_ready=rst_n&&!clear&&(state==IDLE)&&!m_valid;
    always @(posedge clk) begin
        if(!rst_n||clear) begin
            state<=IDLE;m_valid<=0;m_data<=0;steps<=0;
            dividend<=0;divisor<=0;quotient<=0;remainder<=0;exponent_q<=0;sign_q<=0;
            round_up<=0;rounded_q<=0;
        end else begin
            if(m_valid&&m_ready)m_valid<=0;
            if(s_valid&&s_ready) begin
                sign_q<=input_value[31];
                if((&input_value[30:23])&&(|input_value[22:0]))begin m_valid<=1;m_data<=32'h7fc00000;end
                else if(&input_value[30:23])begin m_valid<=1;m_data<={input_value[31],31'b0};end
                else if(input_value[30:0]==0)begin m_valid<=1;m_data<={input_value[31],8'hff,23'b0};end
                else if(input_value[22:0]==0)begin
                    m_valid<=1;
                    // Exact powers of two cannot round across the FTZ edge.
                    m_data<=power_exponent==0 ? {input_value[31],31'b0} : {input_value[31],power_exponent[7:0],23'b0};
                end else begin
                    state<=DIVIDE;steps<=6'd48;dividend<=48'h800000000000;
                    divisor<={1'b1,input_value[22:0]};quotient<=0;remainder<=0;
                    exponent_q<=10'sd253-$signed({2'b0,input_value[30:23]});
                end
            end
            case(state)
                DIVIDE:begin
                    dividend<={dividend[46:0],1'b0};quotient<=quotient_next;remainder<=remainder_next;steps<=steps-1'b1;
                    if(steps==1)state<=COMPARE;
                end
                COMPARE:begin
                    round_up<=(twice_remainder>{2'b0,divisor})||
                        ((twice_remainder=={2'b0,divisor})&&quotient[0]);
                    state<=ROUND;
                end
                ROUND:begin
                    rounded_q<={1'b0,quotient}+round_up;
                    state<=PACK;
                end
                PACK:begin
                    state<=IDLE;m_valid<=1;
                    if(packed_exponent>=10'sd255)m_data<={sign_q,8'hff,23'b0};
                    // For normal binary32 operands with a nonzero mantissa,
                    // e<=0 is at least one subnormal ULP below min-normal.
                    // Its reciprocal cannot round back into the normal range.
                    else if(packed_exponent<=0)m_data<={sign_q,31'b0};
                    else m_data<={sign_q,packed_exponent[7:0],packed_mantissa[22:0]};
                end
                default:begin end
            endcase
        end
    end
endmodule
