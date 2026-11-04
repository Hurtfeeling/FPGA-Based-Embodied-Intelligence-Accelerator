`timescale 1ns/1ps
// Signed INT8 output-stationary systolic array.
// A enters each row from the left; B enters each column from the top.
// Inputs at one accepted advance carry the same K index. Internal skew aligns
// them at every PE. The registered multiplier adds one accepted advance cycle:
// after the last valid input, 2*N-1 advance cycles drain all MACs.
module gemm_systolic_array #(
    parameter integer N = 32,
    parameter integer RESULT_PIPELINE = 0,
    parameter integer LOCAL_CONTROL = 0,
    parameter integer PRE_SKEWED = 0,
    parameter integer ROW_W = (N <= 1) ? 1 : $clog2(N)
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear,
    input  wire                 advance,
    input  wire                 in_valid,
    // With LOCAL_CONTROL=1 these predict the NEXT clock's accepted phase.
    // The scalar advance/in_valid interface remains available for unit use.
    input  wire                 advance_next,
    input  wire                 in_valid_next,
    input  wire [N*8-1:0]       a_vec,
    input  wire [N*8-1:0]       b_vec,
    input  wire [N-1:0]         a_lane_valid,b_lane_valid,
    input  wire [ROW_W-1:0]     result_row,
    output wire [N*32-1:0]      result_data
);
    // Only entries below the diagonal are used; unused entries are pruned.
    reg signed [7:0] a_skew [0:N*N-1];
    reg signed [7:0] b_skew [0:N*N-1];
    reg a_skew_valid [0:N*N-1];
    reg b_skew_valid [0:N*N-1];
    reg signed [7:0] a_forward [0:N*N-1];
    reg signed [7:0] b_forward [0:N*N-1];
    reg a_forward_valid [0:N*N-1];
    reg b_forward_valid [0:N*N-1];
    wire signed [31:0] accum [0:N*N-1];

    localparam integer TILE_SIDE=8,TILES=(N+TILE_SIDE-1)/TILE_SIDE;
    localparam integer GROUPS=TILES*TILES;
    wire [GROUPS-1:0] local_advance,local_valid;
    genvar cg;
    generate for(cg=0;cg<GROUPS;cg=cg+1)begin:g_control
        if(LOCAL_CONTROL)begin:g_registered
            // Preserve the small per-tile replicas, not the payload/PEs.
            (* dont_touch="yes" *) reg advance_q,valid_q;
            always @(posedge clk)begin
                if(!rst_n||clear)begin advance_q<=0;valid_q<=0;end
                else begin advance_q<=advance_next;valid_q<=in_valid_next;end
            end
            assign local_advance[cg]=advance_q;
            assign local_valid[cg]=valid_q;
        end else begin:g_direct
            assign local_advance[cg]=advance;
            assign local_valid[cg]=in_valid;
        end
    end endgenerate

    genvar r, c, d;
    generate
        for (r = 1; r < N; r = r + 1) begin : g_skew_lane
            for (d = 0; d < r; d = d + 1) begin : g_delay
              if(!PRE_SKEWED)begin:g_internal_skew
                // Stale payload is harmless after its valid bit is cleared.
                // Keep reset/clear out of the byte-wide delay data paths.
                always @(posedge clk) if (local_advance[(r/TILE_SIDE)*TILES]) begin
                    if (d == 0) begin
                        a_skew[r*N+d] <= a_vec[r*8 +: 8];
                    end else begin
                        a_skew[r*N+d] <= a_skew[r*N+d-1];
                    end
                end
                always @(posedge clk) if (local_advance[r/TILE_SIDE]) begin
                    if (d == 0) begin
                        b_skew[r*N+d] <= b_vec[r*8 +: 8];
                    end else begin
                        b_skew[r*N+d] <= b_skew[r*N+d-1];
                    end
                end
                always @(posedge clk) begin
                    if (!rst_n || clear) begin
                        a_skew_valid[(r)*N+(d)] <= 1'b0;
                    end else if (local_advance[(r/TILE_SIDE)*TILES]) begin
                        if (d == 0) begin
                            a_skew_valid[(r)*N+(d)] <= local_valid[(r/TILE_SIDE)*TILES];
                        end else begin
                            a_skew_valid[(r)*N+(d)] <= a_skew_valid[(r)*N+(d-1)];
                        end
                    end
                end
                always @(posedge clk) begin
                    if (!rst_n || clear) begin
                        b_skew_valid[r*N+d] <= 1'b0;
                    end else if (local_advance[r/TILE_SIDE]) begin
                        if (d == 0) begin
                            b_skew_valid[r*N+d] <= local_valid[r/TILE_SIDE];
                        end else begin
                            b_skew_valid[(r)*N+(d)] <= b_skew_valid[(r)*N+(d-1)];
                        end
                    end
                end
              end
            end
        end
        for (r = 0; r < N; r = r + 1) begin : g_row
            for (c = 0; c < N; c = c + 1) begin : g_pe
                wire signed [7:0] a_in;
                wire signed [7:0] b_in;
                wire a_valid_in;
                wire b_valid_in;
                gemm_pe_mac u_mac(
                    .clk(clk),.rst_n(rst_n),.clear(clear),
                    .advance(local_advance[(r/TILE_SIDE)*TILES+c/TILE_SIDE]),
                    .a(a_in),.b(b_in),.in_valid(a_valid_in && b_valid_in),
                    .result(accum[(r)*N+(c)]));
                if (c == 0) begin : g_a_edge
                    if(PRE_SKEWED)begin:g_pre_skewed
                        assign a_in=a_vec[r*8+:8];
                        assign a_valid_in=a_lane_valid[r];
                    end else if (r == 0) begin : g_direct
                        assign a_in = a_vec[0 +: 8];
                        assign a_valid_in = local_valid[0];
                    end else begin : g_skew
                        assign a_in = a_skew[(r)*N+(r-1)];
                        assign a_valid_in = a_skew_valid[(r)*N+(r-1)];
                    end
                end else begin : g_a_link
                    assign a_in = a_forward[(r)*N+(c-1)];
                    assign a_valid_in = a_forward_valid[(r)*N+(c-1)];
                end
                if (r == 0) begin : g_b_edge
                    if(PRE_SKEWED)begin:g_pre_skewed
                        assign b_in=b_vec[c*8+:8];
                        assign b_valid_in=b_lane_valid[c];
                    end else if (c == 0) begin : g_direct
                        assign b_in = b_vec[0 +: 8];
                        assign b_valid_in = local_valid[0];
                    end else begin : g_skew
                        assign b_in = b_skew[(c)*N+(c-1)];
                        assign b_valid_in = b_skew_valid[(c)*N+(c-1)];
                    end
                end else begin : g_b_link
                    assign b_in = b_forward[(r-1)*N+(c)];
                    assign b_valid_in = b_forward_valid[(r-1)*N+(c)];
                end
                always @(posedge clk) if (local_advance[(r/TILE_SIDE)*TILES+c/TILE_SIDE]) begin
                    a_forward[r*N+c] <= a_in;
                    b_forward[r*N+c] <= b_in;
                end
                always @(posedge clk) begin
                    if (!rst_n || clear) begin
                        a_forward_valid[(r)*N+(c)] <= 1'b0;
                        b_forward_valid[(r)*N+(c)] <= 1'b0;
                    end else if (local_advance[(r/TILE_SIDE)*TILES+c/TILE_SIDE]) begin
                        a_forward_valid[(r)*N+(c)] <= a_valid_in;
                        b_forward_valid[(r)*N+(c)] <= b_valid_in;
                    end
                end
            end
        end
        if(RESULT_PIPELINE && N==32)begin:g_result_pipe
            // Local 8-row selection, then registers, then a 4-way group mux.
            // Caller holds result_row for a capture cycle before storing.
            // No data reset: the consumer's store enable qualifies the result.
            reg [N*32-1:0] group_q[0:3];
            reg [1:0] group_sel_q;
            integer rg,rc;
            always @(posedge clk)begin
                group_sel_q<=result_row[4:3];
                for(rg=0;rg<4;rg=rg+1)
                    for(rc=0;rc<N;rc=rc+1)
                        group_q[rg][rc*32+:32]<=accum[(rg*8+result_row[2:0])*N+rc];
            end
            assign result_data=group_q[group_sel_q];
        end else begin:g_result_comb
            for (c = 0; c < N; c = c + 1) begin : g_result
                assign result_data[c*32 +: 32] = accum[(result_row)*N+(c)];
            end
        end
    endgenerate
endmodule

// One PE: DSP multiplier followed by a signed 32-bit fabric accumulator.
// Scope USE_DSP to each result register so the adder does not consume a DSP.
// With K <= 16352, every legal INT8 dot product fits signed INT32.
// A stalled advance freezes both stages, including the pending valid bit.
module gemm_pe_mac(
    input wire clk,rst_n,clear,advance,
    input wire signed [7:0] a,b,input wire in_valid,
    output wire [31:0] result
);
    (* use_dsp = "yes" *) reg signed [15:0] product_q;
    (* use_dsp = "no" *) reg signed [31:0] accum_q;
    reg product_valid;
    assign result=accum_q;
    // product_valid discards old/unknown products on reset and clear.
    always @(posedge clk) if(advance) product_q<=a*b;
    always @(posedge clk)begin
        if(!rst_n || clear)begin
            product_valid<=0;accum_q<=0;
        end else if(advance)begin
            product_valid<=in_valid;
            if(product_valid)accum_q<=accum_q+{{16{product_q[15]}},product_q};
        end
    end
endmodule
