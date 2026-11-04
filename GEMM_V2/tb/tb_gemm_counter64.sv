`timescale 1ns/1ps
module tb_gemm_counter64;
    reg clk=0;always #5 clk=~clk;reg rst_n=0;
    reg [191:0] cfg_flat=0;wire [479:0] ro_flat;
    reg start_pulse=0,clear_status_pulse=0;wire busy,irq;
    wire dma_cmd_valid;reg dma_cmd_ready=0;wire [31:0] dma_cmd_addr,dma_cmd_bytes;
    reg dma_done=0,dma_error=0,dma_busy=0;wire dma_rx_ready,dma_tx_valid;
    wire [127:0] dma_tx_data;wire wr_cmd_valid;wire [31:0] wr_cmd_addr,wr_cmd_bytes;
    reg wr_busy=0,bus_read_beat=0,bus_write_beat=0;
    gemm_core dut(.clk(clk),.rst_n(rst_n),.cfg_flat(cfg_flat),.start_pulse(start_pulse),.clear_status_pulse(clear_status_pulse),
        .busy(busy),.irq(irq),.ro_flat(ro_flat),.dma_cmd_valid(dma_cmd_valid),.dma_cmd_ready(dma_cmd_ready),
        .dma_cmd_addr(dma_cmd_addr),.dma_cmd_bytes(dma_cmd_bytes),.dma_done(dma_done),.dma_error(dma_error),.dma_busy(dma_busy),
        .dma_rx_data(128'd0),.dma_rx_valid(1'b0),.dma_rx_ready(dma_rx_ready),
        .dma_tx_data(dma_tx_data),.dma_tx_valid(dma_tx_valid),.dma_tx_ready(1'b0),
        .wr_cmd_valid(wr_cmd_valid),.wr_cmd_ready(1'b0),.wr_cmd_addr(wr_cmd_addr),.wr_cmd_bytes(wr_cmd_bytes),
        .wr_done(1'b0),.wr_error(1'b0),.wr_busy(wr_busy),.bus_read_beat(bus_read_beat),.bus_write_beat(bus_write_beat));
    reg [63:0] expected;
    integer k;
    task automatic run_boundary(input [63:0] seed);
        begin
            @(negedge clk);dma_busy=1;dut.state=dut.FAIL;dut.task_active=0;
            dut.cycles_q=seed;dut.cycles_carry_q={(&seed[47:0]),(&seed[31:0]),(&seed[15:0])};expected=seed;
            for(k=0;k<40;k=k+1)begin
                @(negedge clk);dut.task_active=(k%7!=3);
                @(posedge clk);if(k%7!=3)expected=expected+64'd1;
                #1;if(dut.cycles_q!==expected)$fatal(1,"counter rollover/stall seed%h cycle%0d got%h expected%h",seed,k,dut.cycles_q,expected);
            end
        end
    endtask
    initial begin
        repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
        run_boundary(64'h000000000000fffe);
        run_boundary(64'h00000000fffffffe);
        run_boundary(64'h0000fffffffffffe);
        run_boundary(64'hfffffffffffffffe);
        @(negedge clk);rst_n=0;@(posedge clk);#1;
        if(dut.cycles_q!==0)$fatal(1,"counter reset failed");
        $display("PASS segmented64 counter: 16/32/48/64-bit rollover, stalls and reset");$finish;
    end
    initial begin #100000;$fatal(1,"counter timeout");end
endmodule
