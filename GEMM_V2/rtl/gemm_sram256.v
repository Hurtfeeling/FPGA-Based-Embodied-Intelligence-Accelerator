`timescale 1ns/1ps
// 256-bit synchronous SRAM and elastic stream adapters for the GEMM datapath.
module gemm_sram256 #(
    parameter integer DEPTH = 512,
    parameter integer ADDR_W = $clog2(DEPTH)
)(
    input wire clk, input wire en, input wire we,
    input wire [ADDR_W-1:0] addr,
    input wire [255:0] wdata,
    output reg [255:0] rdata
);
    (* ram_style = "block" *) reg [255:0] mem [0:DEPTH-1];
    // No array reset: mapping to physical SRAM/BRAM is intentional.
    always @(posedge clk) if (en) begin
        if (we) mem[addr] <= wdata;
        else rdata <= mem[addr];
    end
endmodule

// Independent 1R1W operand cache. Small unit users retain a one-cycle
// memory; the V2 core selects the three-cycle mixed BRAM/URAM cache below.
// Banks form one logical word-wide port: exactly one bank reads per clock.
module gemm_bt_sram256 #(
    parameter integer DEPTH = 512,
    parameter integer ADDR_W = $clog2(DEPTH),
    parameter integer PIPELINED_READ = 0,
    parameter integer MIXED_RAM = 0
)(
    input wire clk, input wire we, ren,
    input wire [ADDR_W-1:0] waddr, raddr,
    input wire [255:0] wdata,
    output wire [255:0] rdata
);
    generate if(!PIPELINED_READ)begin:g_simple
        (* ram_style="block" *) reg [255:0] mem[0:DEPTH-1];
        reg [255:0] data_q;
        always @(posedge clk)begin
            if(we)mem[waddr]<=wdata;
            if(ren)data_q<=mem[raddr];
        end
        assign rdata=data_q;
    end else if(MIXED_RAM)begin:g_mixed
        // V2 每个 A/BT 缓存总容量 512 KiB，低半 256 KiB 使用 BRAM，
        // 高半 256 KiB 使用 URAM。按逻辑容量各占一半，而非原语个数各一半。
        // BRAM 拆成 4 个 2048x256 bank；URAM 拆成 2 个 4096x256 bank，
        // 避免把原生 4096 深度的 URAM 配成只有一半有效的 bank。
        wire [255:0] bank_data[0:5];
        reg [2:0] select_q,select_d,return_select;
        reg [1:0] valid_q;
        genvar bram_bank,uram_bank;
        for(bram_bank=0;bram_bank<4;bram_bank=bram_bank+1)begin:g_bram
            (* ram_style="block" *) reg [255:0] mem[0:2047];
            (* dont_touch="yes" *) reg read_en_q;
            (* dont_touch="yes" *) reg [10:0] read_addr_q;
            reg [255:0] data_q,pipeline_q;
            always @(posedge clk)begin
                read_en_q<=ren&&!raddr[13]&&(raddr[12:11]==bram_bank);
                read_addr_q<=raddr[10:0];
                if(we&&!waddr[13]&&(waddr[12:11]==bram_bank))mem[waddr[10:0]]<=wdata;
                if(read_en_q)data_q<=mem[read_addr_q];
                pipeline_q<=data_q;
            end
            assign bank_data[bram_bank]=pipeline_q;
        end
        for(uram_bank=0;uram_bank<2;uram_bank=uram_bank+1)begin:g_uram
            (* ram_style="ultra" *) reg [255:0] mem[0:4095];
            (* dont_touch="yes" *) reg read_en_q;
            (* dont_touch="yes" *) reg [11:0] read_addr_q;
            reg [255:0] data_q,pipeline_q;
            always @(posedge clk)begin
                read_en_q<=ren&&raddr[13]&&(raddr[12]==uram_bank);
                read_addr_q<=raddr[11:0];
                if(we&&waddr[13]&&(waddr[12]==uram_bank))mem[waddr[11:0]]<=wdata;
                if(read_en_q)data_q<=mem[read_addr_q];
                // 紧邻 URAM 的输出级，供 Vivado 吸收到内部 OREG。
                pipeline_q<=data_q;
            end
            assign bank_data[4+uram_bank]=pipeline_q;
        end
        always @(posedge clk)begin
            // E0 请求寄存，E1 RAM 读出，E2 输出流水：所有 bank 统一 3 拍。
            valid_q<={valid_q[0],ren};
            select_q<=raddr[13]?{2'b10,raddr[12]}:{1'b0,raddr[12:11]};
            select_d<=select_q;
            if(valid_q[1])return_select<=select_d;
        end
        assign rdata=bank_data[return_select];
        initial if(DEPTH!=16384||ADDR_W!=14)$fatal(1,"mixed cache requires 16384x256");
    end else begin:g_sliced
        // Register the request locally, read a short RAM, then register the
        // eight-way return selection. No reset on RAM or payload registers.
        wire [255:0] slice_data[0:7];
        reg [2:0] select_q,select_d;
        reg [1:0] valid_q;
        reg [255:0] return_q;
        genvar bank;
        for(bank=0;bank<8;bank=bank+1)begin:g_slice
            (* ram_style="block" *) reg [255:0] mem[0:2047];
            (* dont_touch="yes" *) reg read_en_q;
            (* dont_touch="yes" *) reg [10:0] read_addr_q;
            reg [255:0] data_q;
            always @(posedge clk)begin
                read_en_q<=ren&&(raddr[13:11]==bank);
                read_addr_q<=raddr[10:0];
                if(we&&waddr[13:11]==bank)mem[waddr[10:0]]<=wdata;
                if(read_en_q)data_q<=mem[read_addr_q];
            end
            assign slice_data[bank]=data_q;
        end
        always @(posedge clk)begin
            valid_q<={valid_q[0],ren};
            select_q<=raddr[13:11];select_d<=select_q;
            if(valid_q[1])return_q<=slice_data[select_d];
        end
        assign rdata=return_q;
        initial if(DEPTH!=16384||ADDR_W!=14)$fatal(1,"pipelined cache requires 16384x256");
    end endgenerate
endmodule

// One 256-bit read per operand per clock. Requests visit row0..row31,
// then the next 32 K bytes of row0. Each lane is a 32-byte parallel-push,
// byte-pop FIFO: the refill coincides with consumption of its last old byte.
// Its staggered startup supplies the array's row/column skew directly.
module gemm_input_stream256 #(parameter integer READ_LATENCY=1)(
    input wire clk,rst_n,clear,cancel,start,advance,
    input wire [8:0] k_words,
    input wire [13:0] a_base,bt_base,
    output wire read_valid,
    output reg [13:0] a_addr,bt_addr,
    input wire [255:0] a_data,bt_data,
    output wire [255:0] a_vec,bt_vec,
    output wire [31:0] a_valid,bt_valid
);
    reg read_active;
    reg [READ_LATENCY-1:0] response_valid_q;
    reg [4:0] response_lane_q[0:READ_LATENCY-1];
    reg response_last_q[0:READ_LATENCY-1];
    wire response_valid=response_valid_q[READ_LATENCY-1];
    wire [4:0] response_lane=response_lane_q[READ_LATENCY-1];
    wire response_last=response_last_q[READ_LATENCY-1];
    reg [4:0] read_lane;
    reg [8:0] read_word;
    reg [13:0] a_word_base,bt_word_base;
    reg [255:0] a_payload[0:31],bt_payload[0:31];
    reg [5:0] lane_count[0:31];
    reg lane_last[0:31];
    // The request is already registered. Cancellation clears its token at
    // the clock edge instead of routing global error logic into every BRAM EN.
    assign read_valid=read_active;
    integer q,rp;
    always @(posedge clk)begin
        if(!rst_n||clear||cancel)begin
            read_active<=0;response_valid_q<=0;read_lane<=0;
            read_word<=0;a_addr<=0;bt_addr<=0;
            a_word_base<=0;bt_word_base<=0;
            for(q=0;q<32;q=q+1)begin lane_count[q]<=0;lane_last[q]<=0;end
        end else begin
            response_valid_q[0]<=read_valid;
            for(rp=1;rp<READ_LATENCY;rp=rp+1)begin
                response_valid_q[rp]<=response_valid_q[rp-1];
                response_lane_q[rp]<=response_lane_q[rp-1];
                response_last_q[rp]<=response_last_q[rp-1];
            end
            if(read_valid)begin
                response_lane_q[0]<=read_lane;response_last_q[0]<=({1'b0,read_word}+10'd1>={1'b0,k_words});
                if(read_lane==31)begin
                    read_lane<=0;read_word<=read_word+1'b1;
                    a_word_base<=a_word_base+1'b1;bt_word_base<=bt_word_base+1'b1;
                    a_addr<=a_word_base+1'b1;bt_addr<=bt_word_base+1'b1;
                    if({1'b0,read_word}+10'd1>={1'b0,k_words})read_active<=0;
                end else begin
                    read_lane<=read_lane+1'b1;
                    a_addr<=a_addr+{5'd0,k_words};bt_addr<=bt_addr+{5'd0,k_words};
                end
            end
            for(q=0;q<32;q=q+1)begin
                if(advance&&lane_count[q]!=0)lane_count[q]<=lane_count[q]-1'b1;
                if(response_valid&&response_lane==q)begin
                    lane_count[q]<=32;lane_last[q]<=response_last;
                end
            end
            if(start)begin
                read_active<=1;response_valid_q<=0;read_lane<=0;read_word<=0;
                a_word_base<=a_base;bt_word_base<=bt_base;a_addr<=a_base;bt_addr<=bt_base;
                for(q=0;q<32;q=q+1)begin lane_count[q]<=0;lane_last[q]<=0;end
            end
        end
    end
    genvar lane;
    generate for(lane=0;lane<32;lane=lane+1)begin:g_lane
        // A fault may clock one stale payload; the synchronous count/valid
        // clear discards it. Payload enables contain no global error cone.
        wire push=response_valid&&(response_lane==lane);
        wire pop=advance&&(lane_count[lane]!=0);
        assign a_vec[lane*8+:8]=a_payload[lane][7:0];
        assign bt_vec[lane*8+:8]=bt_payload[lane][7:0];
        assign a_valid[lane]=(lane_count[lane]!=0);
        assign bt_valid[lane]=a_valid[lane];
        // Payload is never reset; count/valid discard old contents.
        always @(posedge clk)begin
            if(push)begin a_payload[lane]<=a_data;bt_payload[lane]<=bt_data;end
            else if(pop)begin a_payload[lane]<=a_payload[lane]>>8;bt_payload[lane]<=bt_payload[lane]>>8;end
        end
`ifndef SYNTHESIS
        always @(posedge clk)if(rst_n&&!clear&&!cancel)begin
            if(push&&(lane_count[lane]>(pop?1:0)))
                $fatal(1,"input lane FIFO overflow lane=%0d",lane);
            if(pop&&lane_count[lane]==1&&!lane_last[lane]&&!push)
                $fatal(1,"input lane FIFO refill missed lane=%0d",lane);
        end
`endif
    end endgenerate
endmodule

module gemm_pack128_256(
    input wire clk, input wire rst_n, input wire flush,
    input wire [127:0] s_data, input wire s_valid, output wire s_ready,
    output reg [255:0] m_data, output reg m_valid, input wire m_ready,
    output wire empty
);
    reg [127:0] low_q;
    reg half_q;
    assign s_ready = !half_q || !m_valid || m_ready;
    assign empty = !half_q && !m_valid;
    // Payload has no reset/flush path; half_q and m_valid discard stale data.
    always @(posedge clk) if (s_valid && s_ready) begin
        if (!half_q) low_q <= s_data;
        else m_data <= {s_data,low_q};
    end
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            half_q <= 0; m_valid <= 0;
        end else begin
            if (m_valid && m_ready) m_valid <= 0;
            if (s_valid && s_ready) begin
                if (!half_q) half_q <= 1;
                else begin m_valid <= 1; half_q <= 0; end
            end
        end
    end
endmodule

module gemm_unpack256_128(
    input wire clk, input wire rst_n, input wire flush,
    input wire [255:0] s_data, input wire s_valid, output wire s_ready,
    output wire [127:0] m_data, output wire m_valid, input wire m_ready,
    output wire empty
);
    reg [255:0] data_q;
    reg valid_q, high_q;
    assign s_ready = !valid_q || (m_ready && high_q);
    assign m_data = high_q ? data_q[255:128] : data_q[127:0];
    assign m_valid = valid_q;
    assign empty = !valid_q;
    // data_q is meaningful only while valid_q is asserted.
    always @(posedge clk) if (s_valid && s_ready) data_q <= s_data;
    always @(posedge clk) begin
        if (!rst_n || flush) begin valid_q <= 0; high_q <= 0; end
        else if (s_valid && s_ready) begin valid_q <= 1; high_q <= 0; end
        else if (valid_q && m_ready) begin
            if (high_q) begin valid_q <= 0; high_q <= 0; end
            else high_q <= 1;
        end
    end
endmodule

// Output C memory: 4096x512 bits = 256 KiB, independent 1W1R.
// Two-cycle read; payload and read pipeline have no reset.
module gemm_result_sram512 #(parameter RAM_STYLE="block")(
    input wire clk,we,ren,
    input wire [11:0] waddr,raddr,
    input wire [511:0] wdata,output reg [511:0] rdata
);
    (* ram_style=RAM_STYLE *) reg [511:0] mem[0:4095];
    reg [511:0] read_q;
    reg read_valid_q;
    always @(posedge clk)begin
        if(we)mem[waddr]<=wdata;
        if(ren)read_q<=mem[raddr];
        read_valid_q<=ren;
        if(read_valid_q)rdata<=read_q;
    end
endmodule

// Four AXI128 beats per C word. Reset only validity and the quarter index.
module gemm_unpack512_128(
    input wire clk,rst_n,flush,
    input wire [511:0] s_data,input wire s_valid,output wire s_ready,
    output wire [127:0] m_data,output wire m_valid,input wire m_ready,
    output wire empty
);
    reg [511:0] data_q;
    reg valid_q;
    reg [1:0] part_q;
    assign s_ready=!valid_q||(m_ready&&part_q==3);
    assign m_data=data_q[part_q*128+:128];
    assign m_valid=valid_q;
    assign empty=!valid_q;
    always @(posedge clk)if(s_valid&&s_ready)data_q<=s_data;
    always @(posedge clk)begin
        if(!rst_n||flush)begin valid_q<=0;part_q<=0;end
        else if(s_valid&&s_ready)begin valid_q<=1;part_q<=0;end
        else if(valid_q&&m_ready)begin
            part_q<=part_q+1'b1;
            if(part_q==3)valid_q<=0;
        end
    end
endmodule
