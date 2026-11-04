`timescale 1ns/1ps
// Python-generated independent results are compared at every output transfer.
// Payload/tag/keep must remain stable during stalls and clear cancels tokens.
module tb_softmax_mul4;
    parameter integer N_BEATS=4096;
    reg clk=0;always #2 clk=~clk;
    reg rst_n=0,clear=0,s_valid=0,m_ready=0;
    wire s_ready,m_valid;
    reg [127:0] s_data=0;
    reg [31:0] s_scale=0;
    reg [3:0] s_keep=0;
    reg [9:0] s_tag=0;
    wire [127:0] m_data;
    wire [3:0] m_keep;
    wire [9:0] m_tag;
    reg [291:0] vectors[0:N_BEATS-1];
    softmax_fp32_mul4 dut(
        .clk(clk),.rst_n(rst_n),.clear(clear),
        .s_valid(s_valid),.s_ready(s_ready),.s_data(s_data),.s_scale(s_scale),
        .s_keep(s_keep),.s_tag(s_tag),.m_valid(m_valid),.m_ready(m_ready),
        .m_data(m_data),.m_keep(m_keep),.m_tag(m_tag));

    integer sent=0,received=0,index,phase=0,run=0,max_run=0,stalls=0;
    reg stalled=0;
    reg [127:0] held_data;
    reg [3:0] held_keep;
    reg [9:0] held_tag;
    reg [31:0] lfsr=32'h250f0032;
    always @(negedge clk) begin
        lfsr={lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
        // First 64 beats prove sustained II=1. Thereafter use reproducible
        // output stalls while the source also inserts occasional bubbles.
        if(phase==1) m_ready=(received<64)||(lfsr[1:0]!=0);
        else m_ready=0;
    end
    always @(posedge clk) begin
        if(!rst_n||clear) begin stalled<=0;run=0;end
        else begin
            if(stalled&&(!m_valid||m_data!==held_data||m_keep!==held_keep||m_tag!==held_tag))
                $fatal(1,"mul4 changed payload, keep or tag under backpressure");
            stalled<=m_valid&&!m_ready;
            held_data<=m_data;held_keep<=m_keep;held_tag<=m_tag;
            if(phase==1&&s_valid&&s_ready) begin
                sent=sent+1;run=run+1;
                if(run>max_run) max_run=run;
            end else run=0;
            if(phase==1&&m_valid&&!m_ready) stalls=stalls+1;
            if(phase==1&&m_valid&&m_ready) begin
                if(received>=N_BEATS||received>=sent)
                    $fatal(1,"mul4 emitted an unaccepted or duplicate token");
                if(m_tag!==received[9:0]||m_keep!==vectors[received][291:288]||
                   m_data!==vectors[received][127:0])
                    $fatal(1,"mul4 beat=%0d tag=%0d keep=%h got=%h expected=%h",
                           received,m_tag,m_keep,m_data,vectors[received][127:0]);
                received=received+1;
            end
        end
    end

    initial begin
        $readmemh("Softmax/tb/build/mul4/vectors.mem",vectors);
        repeat(4) @(negedge clk);
        rst_n=1;s_valid=1;s_data={4{32'h3f800000}};s_scale=32'h3f800000;s_keep=15;
        // Fill a stalled pipeline and clear it before the real vector stream.
        repeat(32) @(negedge clk);
        if(s_ready) $fatal(1,"mul4 full stalled pipeline failed to backpressure");
        clear=1;s_valid=0;
        @(negedge clk);clear=0;
        @(negedge clk);
        if(m_valid) $fatal(1,"mul4 clear leaked a token");
        phase=1;
        @(negedge clk);
        for(index=0;index<N_BEATS;index=index+1) begin
            if(index>=64&&index%19==0) begin
                s_valid=0;repeat(1+index%3) @(negedge clk);
            end
            {s_keep,s_scale,s_data}=vectors[index][291:128];
            s_tag=index[9:0];s_valid=1;
            @(posedge clk);while(!s_ready) @(posedge clk);
            @(negedge clk);
        end
        s_valid=0;
        wait(received==N_BEATS);
        // Continue observing briefly to catch duplicated trailing tokens.
        repeat(16) @(negedge clk);
        if(sent!=N_BEATS||max_run<16||stalls==0)
            $fatal(1,"mul4 coverage failed: sent=%0d run=%0d stalls=%0d",sent,max_run,stalls);
        $display("PASS mul4: %0d exact binary32 lanes, DAZ/RNE/FTZ, clear/stalls/tag/keep, II=1",
                 N_BEATS*4);
        $finish;
    end
    initial begin #500000;$fatal(1,"mul4 timeout");end
endmodule
