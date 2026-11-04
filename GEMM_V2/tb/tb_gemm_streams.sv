`timescale 1ns/1ps
module streams_sram_case(output reg done);
    reg clk=0; always #5 clk=~clk;
    reg en=0,we=0; reg [3:0] addr=0; reg [255:0] wdata=0;
    wire [255:0] rdata;
    reg [255:0] held;
    integer i;
    gemm_sram256 #(.DEPTH(16),.ADDR_W(4)) dut(.*);
    function automatic [255:0] data(input integer id);
        integer j; begin for(j=0;j<8;j=j+1) data[j*32 +:32]=32'h5a819027^(id*193+j*47); end
    endfunction
    initial begin
        done=0;
        for(i=0;i<16;i=i+1) begin
            @(negedge clk); en=1; we=1; addr=i; wdata=data(i);
            @(posedge clk); #1;
        end
        for(i=15;i>=0;i=i-1) begin
            @(negedge clk); we=0; addr=i;
            held=rdata;
            #1; if(rdata!==held) $fatal(1,"SRAM read changed before clock");
            @(posedge clk); #1;
            if(rdata!==data(i)) $fatal(1,"SRAM synchronous read mismatch addr=%0d",i);
        end
        held=rdata;
        @(negedge clk); en=0; addr=7; wdata='1; we=1;
        repeat(3) begin @(posedge clk); #1; if(rdata!==held) $fatal(1,"disabled SRAM read did not hold"); end
        @(negedge clk); en=1; we=1; addr=8; wdata=data(99);
        @(posedge clk); #1; if(rdata!==held) $fatal(1,"SRAM write changed read output");
        @(negedge clk); we=0;
        @(posedge clk); #1; if(rdata!==data(99)) $fatal(1,"SRAM overwrite mismatch");
        @(negedge clk); addr=7;
        @(posedge clk); #1; if(rdata!==data(7)) $fatal(1,"disabled SRAM write changed memory");
        $display("PASS SRAM: synchronous read, enable hold, write, overwrite"); done=1;
    end
endmodule

module streams_pack_case(output reg done);
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,flush=0,s_valid=0,m_ready=0;
    reg [127:0] s_data=0;
    wire s_ready,m_valid,empty; wire [255:0] m_data;
    gemm_pack128_256 dut(.*);
    reg [255:0] queue[0:2047];
    reg [127:0] low_ref;
    reg half_ref=0,held_valid=0;
    reg [255:0] held;
    integer h=0,t=0,checked=0,blocked=0,both=0,seq=0,cyc,seed=91;
    function automatic [127:0] beat(input integer id);
        integer j; begin for(j=0;j<4;j=j+1) beat[j*32 +:32]=32'h927ac051^(id*113+j*37); end
    endfunction
    always @(posedge clk) begin
        if(!rst_n || flush) begin h=0;t=0;half_ref=0;held_valid=0; end
        else begin
            if(empty !== ((h==t)&&!half_ref)) $fatal(1,"pack empty mismatch");
            if(held_valid && (!m_valid || m_data!==held)) $fatal(1,"pack failed backpressure hold");
            held_valid=m_valid&&!m_ready; held=m_data;
            if(s_valid&&!s_ready) blocked=blocked+1;
            if(s_valid&&s_ready&&m_valid&&m_ready) both=both+1;
            if(m_valid&&m_ready) begin
                if(h==t || m_data!==queue[h]) $fatal(1,"pack low/high order or data mismatch");
                h=h+1;checked=checked+1;
            end
            if(s_valid&&s_ready) begin
                if(!half_ref) begin low_ref=s_data;half_ref=1;end
                else begin queue[t]={s_data,low_ref};t=t+1;half_ref=0;end
            end
        end
    end
    initial begin
        done=0; repeat(2) @(posedge clk);
        @(negedge clk);rst_n=1;
        for(cyc=0;cyc<650;cyc=cyc+1) begin
            @(negedge clk);
            flush=(cyc==1)||(cyc==20); // Drop a lone half, then a full word plus half.
            if(flush) s_valid=0;
            else if(!s_valid) begin s_valid=(cyc<20)||(($random(seed)&3)!=0);s_data=beat(seq);end
            m_ready=(cyc>20)&&(($random(seed)&7)>2);
            @(posedge clk); #1;
            if(cyc==0 && (m_valid||empty)) $fatal(1,"pack did not retain the first half");
            if((cyc==1||cyc==20) && (!empty||m_valid)) $fatal(1,"pack flush retained data");
        end
        @(negedge clk);m_ready=1;
        while(s_valid) begin @(posedge clk);#1;end
        @(negedge clk);s_valid=0;
        repeat(8) @(posedge clk);
        #1;
        // An odd final input is deliberately discarded, with no padded output.
        @(negedge clk);flush=1;
        @(posedge clk);#1;if(!empty||m_valid) $fatal(1,"pack flush failed");
        if(checked<50||blocked==0||both==0) $fatal(1,"pack coverage insufficient");
        $display("PASS pack: %0d words, both-side stalls, simultaneous transfer, odd flush",checked);done=1;
    end
    // Drive a new beat only after the old one was accepted, while preserving
    // valid/data throughout any stall. Nonblocking source update avoids monitor races.
    always @(posedge clk) if(rst_n&&!flush&&s_valid&&s_ready) begin
        seq<=seq+1;s_valid<=0;
    end
endmodule

module streams_unpack_case(output reg done);
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,flush=0,s_valid=0,m_ready=0;
    reg [255:0] s_data=0;
    wire s_ready,m_valid,empty;wire [127:0] m_data;
    gemm_unpack256_128 dut(.*);
    reg [127:0] queue[0:4095],held;
    reg held_valid=0;
    integer h=0,t=0,checked=0,blocked=0,both=0,seq=0,cyc,seed=217;
    function automatic [255:0] word(input integer id);
        integer j;begin for(j=0;j<8;j=j+1) word[j*32 +:32]=32'h40219aef^(id*199+j*71);end
    endfunction
    always @(posedge clk) begin
        if(!rst_n||flush) begin h=0;t=0;held_valid=0;end
        else begin
            if(empty !== (h==t)) $fatal(1,"unpack empty mismatch");
            if(held_valid&&(!m_valid||m_data!==held)) $fatal(1,"unpack failed hold");
            held_valid=m_valid&&!m_ready;held=m_data;
            if(s_valid&&!s_ready) blocked=blocked+1;
            if(s_valid&&s_ready&&m_valid&&m_ready) both=both+1;
            if(m_valid&&m_ready) begin
                if(h==t||m_data!==queue[h]) $fatal(1,"unpack low/high order mismatch");
                h=h+1;checked=checked+1;
            end
            if(s_valid&&s_ready) begin
                queue[t]=s_data[127:0];queue[t+1]=s_data[255:128];t=t+2;
            end
        end
    end
    initial begin
        done=0;repeat(2) @(posedge clk);@(negedge clk);rst_n=1;
        for(cyc=0;cyc<750;cyc=cyc+1) begin
            @(negedge clk);flush=(cyc==2)||(cyc==20)||(cyc==253);
            if(flush) s_valid=0;
            else if(!s_valid) begin s_valid=(cyc<20)||(($random(seed)&3)!=0);s_data=word(seq);end
            m_ready=(cyc==1)||((cyc>20)&&(($random(seed)&7)>2));
            @(posedge clk);#1;
            if(cyc==1 && !m_valid) $fatal(1,"unpack lost high half before odd flush");
            if(cyc==2 && (!empty||m_valid)) $fatal(1,"unpack odd-half flush failed");
        end
        @(negedge clk);m_ready=1;
        while(s_valid) begin @(posedge clk);#1;end
        @(negedge clk);s_valid=0;
        repeat(8) @(posedge clk);#1;
        if(!empty||h!=t) $fatal(1,"unpack did not drain");
        if(checked<100||blocked==0||both==0) $fatal(1,"unpack coverage insufficient");
        $display("PASS unpack: %0d halves, low-first, stalls, flush, replace on high transfer",checked);done=1;
    end
    always @(posedge clk) if(rst_n&&!flush&&s_valid&&s_ready) begin seq<=seq+1;s_valid<=0;end
endmodule

module tb_gemm_streams;
    wire ds,dp,du,dr,db;
    streams_bt_sram_case tb(db);
    streams_reset_case tr(dr);
    streams_sram_case ts(ds);
    streams_pack_case tp(dp);
    streams_unpack_case tu(du);
    initial begin wait(ds&&dp&&du&&dr&&db);$display("PASS tb_gemm_streams");$finish;end
    initial begin #1000000;$fatal(1,"streams timeout");end
endmodule

module streams_bt_sram_case(output reg done);
    reg clk=0;always #5 clk=~clk;
    reg we=0,ren=0;
    reg [3:0] waddr=0,raddr=0;
    reg [255:0] wdata=0,held;
    wire [255:0] rdata;
    integer i;
    gemm_bt_sram256 #(.DEPTH(16),.ADDR_W(4)) dut(.*);
    function automatic [255:0] data(input integer id);
        integer j;begin for(j=0;j<8;j=j+1)data[j*32+:32]=32'h71c395a0^(id*137+j*29);end
    endfunction
    initial begin
        done=0;
        for(i=0;i<8;i=i+1)begin
            @(negedge clk);we=1;waddr=i;wdata=data(i);
            @(posedge clk);#1;
        end
        for(i=0;i<8;i=i+1)begin
            @(negedge clk);we=1;waddr=i+8;wdata=data(i+8);ren=1;raddr=i;
            held=rdata;#1;if(rdata!==held)$fatal(1,"BT read was asynchronous");
            @(posedge clk);#1;if(rdata!==data(i))$fatal(1,"BT concurrent write corrupted read");
        end
        for(i=8;i<16;i=i+1)begin
            @(negedge clk);we=0;ren=1;raddr=i;
            @(posedge clk);#1;if(rdata!==data(i))$fatal(1,"BT independent port write lost");
        end
        held=rdata;
        @(negedge clk);ren=0;we=1;waddr=0;wdata=data(99);
        repeat(3)begin @(posedge clk);#1;if(rdata!==held)$fatal(1,"BT disabled read failed hold");end
        @(negedge clk);we=0;ren=1;raddr=0;
        @(posedge clk);#1;if(rdata!==data(99))$fatal(1,"BT overwrite mismatch");
        $display("PASS BT SRAM: independent 1R1W, one-cycle read and output hold");done=1;
    end
endmodule

module streams_reset_case(output reg done);
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,s_valid=0;
    reg [127:0] pack_data=128'h1234;
    reg [255:0] unpack_data=256'h5678;
    wire pv,uv,pe,ue;
    wire [255:0] pd;wire [127:0] ud;
    wire pr,ur;
    reg [255:0] held_p;reg [127:0] held_u;
    gemm_pack128_256 p(.clk(clk),.rst_n(rst_n),.flush(1'b0),
        .s_data(pack_data),.s_valid(s_valid),.s_ready(pr),
        .m_data(pd),.m_valid(pv),.m_ready(1'b0),.empty(pe));
    gemm_unpack256_128 u(.clk(clk),.rst_n(rst_n),.flush(1'b0),
        .s_data(unpack_data),.s_valid(s_valid),.s_ready(ur),
        .m_data(ud),.m_valid(uv),.m_ready(1'b0),.empty(ue));
    initial begin
        done=0;repeat(2) @(posedge clk);
        @(negedge clk);rst_n=1;s_valid=1;
        repeat(5) @(posedge clk);#1;
        if(!pv||!uv) $fatal(1,"stream reset setup failed");
        held_p=pd;held_u=ud;
        @(negedge clk);rst_n=0;s_valid=0;#1;
        if(!pv||!uv||pd!==held_p||ud!==held_u)
            $fatal(1,"stream reset was asynchronous");
        @(posedge clk);#1;
        if(pv||uv||!pe||!ue) $fatal(1,"stream synchronous reset failed");
        @(negedge clk);rst_n=1;
        repeat(3) begin
            @(posedge clk);#1;
            if(pv||uv||!pe||!ue)
                $fatal(1,"stale payload became valid after reset");
        end
        // Old payload is deliberately left in storage. Only fresh transfers
        // may assert valid; the new packed word must contain two fresh halves.
        @(negedge clk);s_valid=1;
        pack_data=128'hdeadbeef00112233;
        unpack_data={128'hcafefeed,128'h1122334455667788};
        @(posedge clk);#1;
        @(negedge clk);pack_data=128'h9988776655443322;
        @(posedge clk);#1;
        @(negedge clk);s_valid=0;
        repeat(4) @(posedge clk);#1;
        if(!pv||!uv ||
            pd!=={128'h9988776655443322,128'hdeadbeef00112233} ||
            ud!==128'h1122334455667788)
            $fatal(1,"stream leaked stale data into fresh post-reset transfers");
        $display("PASS stream synchronous reset: metadata only, no stale payload after restart");done=1;
    end
endmodule
