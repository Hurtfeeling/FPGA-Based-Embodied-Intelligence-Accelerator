`timescale 1ns/1ps
// Four-register AXI-Lite control window. AW and W may arrive in either order.
// Protection is checked at commit, including the cycle after START.
// Read payloads are snapshots held until the R handshake.
module softmax_axil_regs #(
    parameter [31:0] C_S_AXI_BASEADDR = 32'h80010000
)(
    input wire busy, input wire done, input wire error,
    input wire [3:0] error_code,
    output reg cfg_enable,
    output reg [31:0] cfg_src, cfg_dst, cfg_dim,
    output reg start_pulse, output reg clear_status_pulse,
    input wire S_AXI_ACLK, input wire S_AXI_ARESETN,
    input wire [31:0] S_AXI_AWADDR, input wire [2:0] S_AXI_AWPROT,
    input wire S_AXI_AWVALID, output wire S_AXI_AWREADY,
    input wire [31:0] S_AXI_WDATA, input wire [3:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID, output wire S_AXI_WREADY,
    output reg [1:0] S_AXI_BRESP, output reg S_AXI_BVALID, input wire S_AXI_BREADY,
    input wire [31:0] S_AXI_ARADDR, input wire [2:0] S_AXI_ARPROT,
    input wire S_AXI_ARVALID, output wire S_AXI_ARREADY,
    output reg [31:0] S_AXI_RDATA, output reg [1:0] S_AXI_RRESP,
    output reg S_AXI_RVALID, input wire S_AXI_RREADY
);
    localparam [1:0] OKAY=2'b00, SLVERR=2'b10;
    reg aw_full, w_full;
    reg [31:0] aw_addr_q, w_data_q;
    reg [3:0] w_strb_q;
    wire [31:0] wr_offset = aw_addr_q-C_S_AXI_BASEADDR;
    wire [31:0] rd_offset = S_AXI_ARADDR-C_S_AXI_BASEADDR;
    wire wr_mapped = aw_addr_q>=C_S_AXI_BASEADDR &&
                     wr_offset<=32'd12 && wr_offset[1:0]==0;
    wire rd_mapped = S_AXI_ARADDR>=C_S_AXI_BASEADDR &&
                     rd_offset<=32'd12 && rd_offset[1:0]==0;
    wire protected_busy = busy || start_pulse;
    wire is_control = wr_offset==0;
    wire request_start = is_control && w_strb_q[1] && w_data_q[8];
    wire request_clear = is_control && w_strb_q[1] && w_data_q[9];
    wire [31:0] status_word = {24'd0,error_code,error,done,busy,cfg_enable};
    reg [31:0] merge_word, write_mask;
    reg write_bad;
    integer b;

    assign S_AXI_AWREADY = S_AXI_ARESETN && !aw_full && !S_AXI_BVALID;
    assign S_AXI_WREADY = S_AXI_ARESETN && !w_full && !S_AXI_BVALID;
    assign S_AXI_ARREADY = S_AXI_ARESETN && !S_AXI_RVALID;

    always @* begin
        case (wr_offset[3:2])
            0: merge_word={31'd0,cfg_enable};
            1: merge_word=cfg_src;
            2: merge_word=cfg_dst;
            3: merge_word=cfg_dim;
            default: merge_word=0;
        endcase
        write_mask=0;
        for (b=0;b<4;b=b+1) begin
            write_mask[b*8+:8]={8{w_strb_q[b]}};
            if (w_strb_q[b]) merge_word[b*8+:8]=w_data_q[b*8+:8];
        end
        write_bad=!wr_mapped;
        if (w_strb_q!=0) begin
            if (protected_busy) write_bad=1;
            if (is_control && ((w_data_q & write_mask & ~32'h00000301)!=0))
                write_bad=1;
            if (request_start && (!merge_word[0] || request_clear))
                write_bad=1;
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            cfg_enable<=0; cfg_src<=0; cfg_dst<=0; cfg_dim<=0;
            aw_full<=0; w_full<=0; aw_addr_q<=0; w_data_q<=0; w_strb_q<=0;
            S_AXI_BRESP<=OKAY; S_AXI_BVALID<=0;
            S_AXI_RDATA<=0; S_AXI_RRESP<=OKAY; S_AXI_RVALID<=0;
            start_pulse<=0; clear_status_pulse<=0;
        end else begin
            start_pulse<=0; clear_status_pulse<=0;
            if (S_AXI_BVALID && S_AXI_BREADY) S_AXI_BVALID<=0;
            if (S_AXI_RVALID && S_AXI_RREADY) S_AXI_RVALID<=0;
            if (S_AXI_AWVALID && S_AXI_AWREADY) begin
                aw_addr_q<=S_AXI_AWADDR; aw_full<=1;
            end
            if (S_AXI_WVALID && S_AXI_WREADY) begin
                w_data_q<=S_AXI_WDATA; w_strb_q<=S_AXI_WSTRB; w_full<=1;
            end
            if (aw_full && w_full && !S_AXI_BVALID) begin
                aw_full<=0; w_full<=0; S_AXI_BVALID<=1;
                S_AXI_BRESP<=write_bad ? SLVERR : OKAY;
                if (!write_bad && w_strb_q!=0) begin
                    case (wr_offset[3:2])
                        0: begin
                            cfg_enable<=merge_word[0];
                            start_pulse<=request_start;
                            clear_status_pulse<=request_clear;
                        end
                        1: cfg_src<=merge_word;
                        2: cfg_dst<=merge_word;
                        3: cfg_dim<=merge_word;
                    endcase
                end
            end
            if (S_AXI_ARVALID && S_AXI_ARREADY) begin
                S_AXI_RVALID<=1; S_AXI_RRESP<=rd_mapped ? OKAY : SLVERR;
                if (!rd_mapped) S_AXI_RDATA<=0;
                else case (rd_offset[3:2])
                    0: S_AXI_RDATA<=status_word;
                    1: S_AXI_RDATA<=cfg_src;
                    2: S_AXI_RDATA<=cfg_dst;
                    3: S_AXI_RDATA<=cfg_dim;
                    default: S_AXI_RDATA<=0;
                endcase
            end
        end
    end
`ifndef SYNTHESIS
    initial begin
        if (C_S_AXI_BASEADDR[1:0]!=0 || C_S_AXI_BASEADDR>32'hfffffff0)
            $fatal(1,"AXI-Lite base must cover the aligned 16-byte window");
    end
`endif
endmodule
