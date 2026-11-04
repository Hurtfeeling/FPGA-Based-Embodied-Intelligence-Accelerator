`timescale 1ns/1ps
module tb_gemm_output_writer512 #(parameter integer HALF_MAJOR=0);
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,cancel_all=0,ctx_valid=0;
    wire ctx_ready;
    reg [31:0] ctx_job=0,ctx_m0=32,ctx_n0=0,ctx_ybase=32'h100000,ctx_ystride=128;
    reg [11:0] ctx_cbase=4032;
    reg [6:0] produced_words=0;
    wire c_req_valid;wire [11:0] c_req_addr;
    reg c_rsp_valid=0;reg [511:0] c_rsp_data=0;
    wire wr_cmd_valid;reg wr_cmd_ready=1;
    wire [31:0] wr_cmd_addr,wr_cmd_bytes,wr_cmd_stride;wire wr_cmd_strided;
    reg wr_done=0,wr_error=0;
    wire [127:0] tx_data;wire tx_valid;reg tx_ready=0;
    wire done,fault,busy;wire [31:0] active_job;
    gemm_output_writer512 #(.C_HALF_MAJOR(HALF_MAJOR)) dut(.*);
    reg req_d=0;reg [511:0] data_d;
    reg command_active=0,fail_first=0,zero_tail=0,fault_seen=0,done_seen=0;
    integer beats_in_cmd=0,cmd_target=0,total_beats=0,commands=0,response_delay=0;
    integer trial,cyc,timeout;
    reg [31:0] rng=32'h8af13455;
    reg [511:0] expword;
    reg full_rate=0,hold_tx=0,stalled_tx=0;
    reg [127:0] held_tx;
    integer w_streak=0,max_w_streak=0;
    function automatic [511:0] word(input integer id);
        integer j;begin for(j=0;j<16;j=j+1)word[j*32+:32]=32'hacf19320^(id*197+j*53);end
    endfunction
    always @(posedge clk)begin
        if(!rst_n)begin req_d<=0;c_rsp_valid<=0;end
        else begin
            req_d<=c_req_valid;c_rsp_valid<=req_d;
            if(c_req_valid)begin
                if(c_req_addr-ctx_cbase>=produced_words)$fatal(1,"read before physical C commit");
                data_d<=word(HALF_MAJOR?((c_req_addr-ctx_cbase)%32)*2+(c_req_addr-ctx_cbase)/32:c_req_addr-ctx_cbase);
            end
            if(req_d)c_rsp_data<=data_d;
        end
        if(rst_n)begin
            rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            wr_done<=0;wr_error<=0;
            if(fault)fault_seen<=1;
            if(done)done_seen<=1;
            if(wr_cmd_valid&&wr_cmd_ready)begin
                if(command_active)$fatal(1,"writer issued overlapping commands");
                if(produced_words==0)$fatal(1,"writer started before first committed word");
                if(wr_cmd_bytes!=4096||commands!=0||wr_cmd_strided!==(ctx_ystride!=128)||wr_cmd_stride!=ctx_ystride)
                    $fatal(1,"microtile command/stride metadata mismatch");
                if(wr_cmd_addr!==ctx_ybase+ctx_m0*ctx_ystride+ctx_n0*4)
                    $fatal(1,"writer DDR first row address");
                command_active<=1;cmd_target<=wr_cmd_bytes/16;beats_in_cmd<=0;response_delay<=0;commands<=commands+1;
            end
            if(tx_valid&&tx_ready)begin
                if(!command_active)$fatal(1,"data outside accepted command");
                expword=(zero_tail&&((HALF_MAJOR?((total_beats/4)%2)*32+(total_beats/4)/2:total_beats/4)>=3))?512'd0:word(total_beats/4);
                if(tx_data!==expword[(total_beats%4)*128+:128])$fatal(1,"512 to128 order/data mismatch beat%0d",total_beats);
                total_beats<=total_beats+1;beats_in_cmd<=beats_in_cmd+1;
            end
            if(command_active&&beats_in_cmd==cmd_target)begin
                if(response_delay==4)begin
                    wr_done<=1;wr_error<=fail_first;command_active<=0;
                end else response_delay<=response_delay+1;
            end
        end
    end
    always @(negedge clk)tx_ready=rst_n&&command_active&&(beats_in_cmd<cmd_target)&&
                                         !hold_tx&&(full_rate||rng[2:0]!=0);
    always @(posedge clk)begin
        if(!rst_n||dut.stream_flush)stalled_tx<=0;
        else begin
            if(stalled_tx&&(!tx_valid||tx_data!==held_tx))
                $fatal(1,"writer lost a held AXI half while DDR was stalled");
            stalled_tx<=tx_valid&&!tx_ready;held_tx<=tx_data;
        end
        if(rst_n&&tx_valid&&tx_ready)begin
            w_streak=w_streak+1;if(w_streak>max_w_streak)max_w_streak=w_streak;
        end else w_streak=0;
    end
    task automatic run(input bit strided,input bit cancel_mid,input bit inject_error);
        begin
            @(negedge clk);ctx_ystride=strided?512:128;ctx_n0=strided?64:0;
            ctx_job=ctx_job+1;ctx_valid=1;produced_words=0;cancel_all=0;
            fail_first=inject_error;zero_tail=cancel_mid;fault_seen=0;done_seen=0;
            total_beats=0;commands=0;
            @(posedge clk);while(!ctx_ready)@(posedge clk);
            @(negedge clk);ctx_valid=0;
            repeat(35)@(negedge clk);
            if(wr_cmd_valid||commands!=0)$fatal(1,"zero produced words accepted by writer");
            produced_words=1;
            repeat(7)@(negedge clk);
            if(commands!=1)$fatal(1,"first 512-bit word did not start writeback");
            if(cancel_mid)begin
                produced_words=3;
                while(total_beats<2)@(negedge clk);
                cancel_all=1;
            end else begin
                for(cyc=2;cyc<=64;cyc=cyc+1)begin @(negedge clk);produced_words=cyc;end
            end
            timeout=0;
            while(busy)begin @(negedge clk);timeout=timeout+1;if(timeout>10000)$fatal(1,"writer timeout");end
            repeat(3)@(negedge clk);
            if(cancel_mid)begin
                if(total_beats!=256||done_seen)$fatal(1,"cancelled command not zero-filled/drained");
            end else if(inject_error)begin
                if(!fault_seen||done_seen)$fatal(1,"write error completion invalid");
            end else if(!done_seen||total_beats!=256||commands!=1)$fatal(1,"writer final counts");
            cancel_all=0;
        end
    endtask
    task automatic dense_stall(input integer stop_after);
        begin
            @(negedge clk);ctx_ystride=128;ctx_n0=0;ctx_job=ctx_job+1;
            ctx_valid=1;produced_words=64;cancel_all=0;
            fail_first=0;zero_tail=0;fault_seen=0;done_seen=0;
            total_beats=0;commands=0;w_streak=0;max_w_streak=0;
            full_rate=1;hold_tx=(stop_after==0);
            @(posedge clk);while(!ctx_ready)@(posedge clk);
            @(negedge clk);ctx_valid=0;
            if(stop_after!=0)begin
                while(total_beats<stop_after)@(negedge clk);
                hold_tx=1;tx_ready=0;
            end
            timeout=0;
            while(!command_active||dut.buffered_words!=3||dut.read_valid_q!=0)begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>200)$fatal(1,"prefetch credits did not fill under DDR stall");
            end
            repeat(30)begin
                @(negedge clk);
                if(c_req_valid||dut.buffered_words!=3)
                    $fatal(1,"writer kept reading C without space during DDR stall");
            end
            hold_tx=0;
            // Avoid a testbench race with the negedge ready driver.
            tx_ready=1;
            timeout=0;
            while(busy)begin
                @(negedge clk);timeout=timeout+1;
                if(timeout>2000)$fatal(1,"dense writer stalled after credit recovery");
            end
            repeat(2)@(negedge clk);
            if(!done_seen||total_beats!=256||commands!=1||max_w_streak!=256-stop_after)
                $fatal(1,"dense writer inserted bubbles or lost words: streak=%0d stop=%0d",max_w_streak,stop_after);
            full_rate=0;
            $display("PASS no-FIFO writer: stall after %0d halves, credit limit3, recovery streak%0d",stop_after,max_w_streak);
        end
    endtask
    initial begin
        repeat(3)@(negedge clk);rst_n=1;
        run(0,0,0);run(1,0,0);run(0,1,0);run(1,0,1);run(0,0,0);
        dense_stall(0);dense_stall(1);dense_stall(2);dense_stall(3);
        $display("PASS writer512: first-word start, strided/contiguous output, stalls, cancellation zero fill, error/recovery");$finish;
    end
    initial begin #1000000;$fatal(1,"writer512 timeout");end
endmodule
