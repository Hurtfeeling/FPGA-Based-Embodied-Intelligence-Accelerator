`timescale 1ns/1ps
// ABI 10: pipelined 32-bit AXI-Lite, absolute address decode, same register map.
module gemm_axil_regs #(
    parameter [31:0] C_S_AXI_BASEADDR=32'h80000000
)(
    input wire busy,
    output wire [191:0] cfg_flat,
    input wire [479:0] ro_flat,
    output reg start_pulse,output reg clear_status_pulse,
    input wire S_AXI_ACLK,input wire S_AXI_ARESETN,
    input wire [31:0] S_AXI_AWADDR,input wire [2:0] S_AXI_AWPROT,
    input wire S_AXI_AWVALID,output wire S_AXI_AWREADY,
    input wire [31:0] S_AXI_WDATA,input wire [3:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID,output wire S_AXI_WREADY,
    output reg [1:0] S_AXI_BRESP,output reg S_AXI_BVALID,input wire S_AXI_BREADY,
    input wire [31:0] S_AXI_ARADDR,input wire [2:0] S_AXI_ARPROT,
    input wire S_AXI_ARVALID,output wire S_AXI_ARREADY,
    output reg [31:0] S_AXI_RDATA,output reg [1:0] S_AXI_RRESP,
    output reg S_AXI_RVALID,input wire S_AXI_RREADY
);
    localparam [1:0] OKAY=0,SLVERR=2;
    reg [191:0] cfg_q;
    reg aw_full,w_full,wr_prepared,rd_pending;
    reg [31:0] aw_addr_q,w_data_q,rd_offset_q;
    reg [3:0] w_strb_q;
    reg rd_mapped_q,wr_mapped_q,wr_ro_q,wr_ctrl_q,wr_k_q;
    reg [2:0] wr_index_q;
    reg [31:0] wr_merged_q;
    reg wr_start_q,wr_clear_q,wr_zero_q;
    wire [31:0] wr_offset=aw_addr_q-C_S_AXI_BASEADDR;
    wire [31:0] rd_offset=S_AXI_ARADDR-C_S_AXI_BASEADDR;
    wire wr_mapped=aw_addr_q>=C_S_AXI_BASEADDR && wr_offset<=32'h50 && wr_offset[1:0]==0;
    wire rd_mapped=S_AXI_ARADDR>=C_S_AXI_BASEADDR && rd_offset<=32'h50 && rd_offset[1:0]==0;
    wire protected_busy=busy||start_pulse;
    assign cfg_flat=cfg_q;
    assign S_AXI_AWREADY=S_AXI_ARESETN&&!aw_full&&!S_AXI_BVALID;
    assign S_AXI_WREADY=S_AXI_ARESETN&&!w_full&&!S_AXI_BVALID;
    assign S_AXI_ARREADY=S_AXI_ARESETN&&!rd_pending&&!S_AXI_RVALID;
    function automatic [31:0] read_rw(input [2:0] index);
        begin case(index)
            0:read_rw={30'd0,cfg_q[1:0]};
            1:read_rw=cfg_q[32+:32];
            2:read_rw=cfg_q[64+:32];
            3:read_rw=cfg_q[96+:32];
            4:read_rw=cfg_q[128+:32];
            5:read_rw=cfg_q[160+:32];
            default:read_rw=0;
        endcase end
    endfunction
    function automatic [31:0] read_word(input [4:0] index);
        begin case(index)
            0:read_word={30'd0,cfg_q[1:0]};
            1:read_word=cfg_q[32+:32];
            2:read_word=cfg_q[64+:32];
            3:read_word=cfg_q[96+:32];
            4:read_word=cfg_q[128+:32];
            5:read_word=cfg_q[160+:32];
            6:read_word=ro_flat[0+:32];
            7:read_word=ro_flat[32+:32];
            8:read_word=ro_flat[64+:32];
            9:read_word=ro_flat[96+:32];
            10:read_word=ro_flat[128+:32];
            11:read_word=ro_flat[160+:32];
            12:read_word=ro_flat[192+:32];
            13:read_word=ro_flat[224+:32];
            14:read_word=ro_flat[256+:32];
            15:read_word=ro_flat[288+:32];
            16:read_word=ro_flat[320+:32];
            17:read_word=ro_flat[352+:32];
            18:read_word=ro_flat[384+:32];
            19:read_word=ro_flat[416+:32];
            20:read_word=ro_flat[448+:32];
            default:read_word=0;
        endcase end
    endfunction
    reg [31:0] merged;
    reg wr_bad;
    integer b;
    always @*begin
        merged=read_rw(wr_offset[4:2]);
        for(b=0;b<4;b=b+1)if(w_strb_q[b])merged[b*8+:8]=w_data_q[b*8+:8];
        // Validation follows the registered merge/decode stage. Dynamic BUSY
        // and ERROR are checked at commit, rather than at AW or W acceptance.
        wr_bad=!wr_mapped_q||wr_ro_q;
        if(protected_busy&&!wr_ctrl_q)wr_bad=1;
        if(wr_ctrl_q&&(wr_merged_q&~32'h303)!=0)wr_bad=1;
        if(wr_k_q&&wr_merged_q[31:16]!=0)wr_bad=1;
        if(wr_start_q&&(protected_busy||ro_flat[2]||!wr_merged_q[0]||wr_clear_q))wr_bad=1;
        if(wr_zero_q)wr_bad=0;
    end
    // Transaction payload is qualified by aw_full/w_full, wr_prepared,
    // rd_pending and RVALID. Only those control tags need synchronous reset.
    always @(posedge S_AXI_ACLK)begin
        if(S_AXI_AWVALID&&S_AXI_AWREADY)aw_addr_q<=S_AXI_AWADDR;
        if(S_AXI_WVALID&&S_AXI_WREADY)begin w_data_q<=S_AXI_WDATA;w_strb_q<=S_AXI_WSTRB;end
        if(S_AXI_ARVALID&&S_AXI_ARREADY)rd_offset_q<=rd_offset;
        if(rd_pending&&!S_AXI_RVALID)
            S_AXI_RDATA<=rd_mapped_q?read_word(rd_offset_q[6:2]):0;
        if(aw_full&&w_full&&!S_AXI_BVALID&&!wr_prepared)wr_merged_q<=merged;
    end
    always @(posedge S_AXI_ACLK)begin
        if(!S_AXI_ARESETN)begin
            cfg_q<=0;aw_full<=0;w_full<=0;wr_prepared<=0;rd_pending<=0;
            rd_mapped_q<=0;wr_mapped_q<=0;wr_ro_q<=0;wr_ctrl_q<=0;wr_k_q<=0;
            wr_index_q<=0;wr_start_q<=0;wr_clear_q<=0;wr_zero_q<=0;
            S_AXI_BRESP<=OKAY;S_AXI_BVALID<=0;S_AXI_RRESP<=OKAY;S_AXI_RVALID<=0;
            start_pulse<=0;clear_status_pulse<=0;
        end else begin
            start_pulse<=0;clear_status_pulse<=0;
            if(S_AXI_BVALID&&S_AXI_BREADY)S_AXI_BVALID<=0;
            if(S_AXI_RVALID&&S_AXI_RREADY)S_AXI_RVALID<=0;
            if(S_AXI_AWVALID&&S_AXI_AWREADY)aw_full<=1;
            if(S_AXI_WVALID&&S_AXI_WREADY)w_full<=1;
            if(S_AXI_ARVALID&&S_AXI_ARREADY)begin
                rd_pending<=1;rd_mapped_q<=rd_mapped;
            end
            if(rd_pending&&!S_AXI_RVALID)begin
                rd_pending<=0;S_AXI_RVALID<=1;S_AXI_RRESP<=rd_mapped_q?OKAY:SLVERR;
            end
            if(aw_full&&w_full&&!S_AXI_BVALID&&!wr_prepared)begin
                wr_prepared<=1;
                wr_mapped_q<=wr_mapped;wr_ro_q<=wr_offset>=32'h18;
                wr_ctrl_q<=wr_offset==0;wr_k_q<=wr_offset==32'h14;
                wr_index_q<=wr_offset[4:2];wr_zero_q<=w_strb_q==0;
                wr_start_q<=(wr_offset==0)&&w_strb_q[1]&&w_data_q[8];
                wr_clear_q<=(wr_offset==0)&&w_strb_q[1]&&w_data_q[9];
            end
            if(wr_prepared&&!S_AXI_BVALID)begin
                wr_prepared<=0;aw_full<=0;w_full<=0;S_AXI_BVALID<=1;
                S_AXI_BRESP<=wr_bad?SLVERR:OKAY;
                if(!wr_bad&&!wr_zero_q)begin
                    if(wr_ctrl_q)begin
                        cfg_q[31:0]<=wr_merged_q&3;
                        start_pulse<=wr_start_q;clear_status_pulse<=wr_clear_q;
                    end else begin
                        case(wr_index_q)
                            1:cfg_q[32+:32]<=wr_merged_q;
                            2:cfg_q[64+:32]<=wr_merged_q;
                            3:cfg_q[96+:32]<=wr_merged_q;
                            4:cfg_q[128+:32]<=wr_merged_q;
                            5:cfg_q[160+:32]<=wr_merged_q;
                            default:begin end
                        endcase
                    end
                end
            end
        end
    end
`ifndef SYNTHESIS
    initial if(C_S_AXI_BASEADDR[1:0]!=0||C_S_AXI_BASEADDR>32'hffffffac)
        $fatal(1,"AXI-Lite BASE must cover the aligned 84-byte ABI 10 window");
`endif
endmodule
