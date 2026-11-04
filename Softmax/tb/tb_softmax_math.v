`timescale 1ns/1ps
module tb_softmax_math;
    `include "softmax_fp32_funcs.vh"
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,clear=0;
    reg [127:0] arith[0:11999];reg [95:0] uq[0:1023];
    reg [163:0] exp_vectors[0:1023],norm_vectors[0:511];reg [63:0] recip_vectors[0:255];
    reg ev=0,nv=0,rv=0;wire er,nr,rr,eo,no,ro;
    reg [127:0] ed=0,nd=0;reg [31:0] em=0,ns=0,rd=0;
    reg [3:0] ek=0,nk=0;reg [9:0] et=0,nt=0;
    wire [127:0] eout,nout;wire [9:0] eot,notag;wire [31:0] rout;
    reg e_ready=0,n_ready=0,r_ready=0;
    softmax_exp4 #(.LUT_FILE("Softmax/rtl/exp_lut.mem")) exp_dut(.clk(clk),.rst_n(rst_n),.clear(clear),.s_valid(ev),.s_ready(er),.s_data(ed),.s_max(em),.s_keep(ek),.s_tag(et),.m_valid(eo),.m_ready(e_ready),.m_data(eout),.m_tag(eot));
    softmax_norm4 norm_dut(.clk(clk),.rst_n(rst_n),.clear(clear),.s_valid(nv),.s_ready(nr),.s_data(nd),.s_scale(ns),.s_keep(nk),.s_tag(nt),.m_valid(no),.m_ready(n_ready),.m_data(nout),.m_tag(notag));
    softmax_recip recip_dut(.clk(clk),.rst_n(rst_n),.clear(clear),.s_valid(rv),.s_ready(rr),.s_data(rd),.m_valid(ro),.m_ready(r_ready),.m_data(rout));
    integer i,phase=0,ecount=0,ncount=0,rcount=0,efd,nfd;
    integer exp_run=0,max_exp_run=0,norm_run=0,max_norm_run=0;
    reg es=0,ns_hold=0,rs=0;reg [127:0] ep,np;reg [9:0] ept,npt;reg [31:0] rp;
    always @(negedge clk) begin
        e_ready=(phase==1)&&(($random & 3)!=0);
        n_ready=(phase==2)&&(($random & 3)!=0);
        r_ready=(phase==3)&&(($random & 3)!=0);
    end
    always @(posedge clk) begin
      if(!rst_n||clear) begin es<=0;ns_hold<=0;rs<=0;exp_run=0;norm_run=0;end
      else begin
        if(es&&(!eo||eout!==ep||eot!==ept)) $fatal(1,"exp changed under backpressure");
        if(ns_hold&&(!no||nout!==np||notag!==npt)) $fatal(1,"norm changed under backpressure");
        if(rs&&(!ro||rout!==rp)) $fatal(1,"recip changed under backpressure");
        es<=eo&&!e_ready;ep<=eout;ept<=eot;
        ns_hold<=no&&!n_ready;np<=nout;npt<=notag;
        rs<=ro&&!r_ready;rp<=rout;
        if(ev&&er&&phase==1) begin exp_run=exp_run+1;if(exp_run>max_exp_run)max_exp_run=exp_run;end
        else exp_run=0;
        if(nv&&nr&&phase==2) begin norm_run=norm_run+1;if(norm_run>max_norm_run)max_norm_run=norm_run;end
        else norm_run=0;
        if(eo&&e_ready) begin $fdisplay(efd,"%03x %032x",eot,eout);ecount=ecount+1;end
        if(no&&n_ready) begin $fdisplay(nfd,"%03x %032x",notag,nout);ncount=ncount+1;end
        if(ro&&r_ready) begin
            if(rout!==recip_vectors[rcount][31:0]) $fatal(1,"recip[%0d] input=%h got=%h expected=%h",rcount,recip_vectors[rcount][63:32],rout,recip_vectors[rcount][31:0]);
            rcount=rcount+1;
        end
      end
    end
    initial begin
        $readmemh("Softmax/tb/build/math/arith.mem",arith);
        $readmemh("Softmax/tb/build/math/uq.mem",uq);
        $readmemh("Softmax/tb/build/math/exp.mem",exp_vectors);
        $readmemh("Softmax/tb/build/math/norm.mem",norm_vectors);
        $readmemh("Softmax/tb/build/math/recip.mem",recip_vectors);
        efd=$fopen("Softmax/tb/build/math/exp_results.txt","w");
        nfd=$fopen("Softmax/tb/build/math/norm_results.txt","w");
        for(i=0;i<12000;i=i+1) begin
            if(fp32_sub(arith[i][127:96],arith[i][95:64])!==arith[i][63:32])
                $fatal(1,"sub[%0d] a=%h b=%h got=%h expected=%h",i,arith[i][127:96],arith[i][95:64],fp32_sub(arith[i][127:96],arith[i][95:64]),arith[i][63:32]);
            if(fp32_mul(arith[i][127:96],arith[i][95:64])!==arith[i][31:0])
                $fatal(1,"mul[%0d] a=%h b=%h got=%h expected=%h",i,arith[i][127:96],arith[i][95:64],fp32_mul(arith[i][127:96],arith[i][95:64]),arith[i][31:0]);
        end
        for(i=0;i<1024;i=i+1)
            if(uq48_to_fp32(uq[i][95:32])!==uq[i][31:0]) $fatal(1,"uq[%0d] got=%h expected=%h",i,uq48_to_fp32(uq[i][95:32]),uq[i][31:0]);
        // Fill stalled pipes, cancel them, and ensure no pre-clear token leaks.
        repeat(3) @(negedge clk);rst_n=1;
        ev=1;ed={4{32'h3f800000}};em=32'h3f800000;ek=15;
        nv=1;nd={4{32'h3f800000}};ns=32'h3f800000;nk=15;
        rv=1;rd=32'h43f00000;
        // Fill the complete subtraction, DSP multiply and LUT pipelines
        // before checking backpressure. Margin accommodates pipeline cuts.
        repeat(64) @(negedge clk);
        if(er||nr||rr) $fatal(1,"stalled pipes did not backpressure");
        clear=1;ev=0;nv=0;rv=0;
        @(negedge clk);clear=0;
        @(negedge clk);
        if(eo||no||ro) $fatal(1,"clear leaked a token");
        phase=1;
        @(negedge clk);
        for(i=0;i<1024;i=i+1) begin
            ev=1;ed=exp_vectors[i][127:0];em=exp_vectors[i][159:128];ek=exp_vectors[i][163:160];et=i;
            @(posedge clk);while(!er) @(posedge clk);
            @(negedge clk);
        end
        ev=0;wait(ecount==1024);@(negedge clk);phase=2;
        @(negedge clk);
        for(i=0;i<512;i=i+1) begin
            nv=1;nd=norm_vectors[i][127:0];ns=norm_vectors[i][159:128];nk=norm_vectors[i][163:160];nt=i;
            @(posedge clk);while(!nr) @(posedge clk);
            @(negedge clk);
        end
        nv=0;wait(ncount==512);@(negedge clk);phase=3;
        @(negedge clk);
        for(i=0;i<256;i=i+1) begin
            rv=1;rd=recip_vectors[i][63:32];
            @(posedge clk);while(!rr) @(posedge clk);
            @(negedge clk);rv=0;
            wait(rcount==i+1);@(negedge clk);
        end
        if(max_exp_run<8||max_norm_run<8) $fatal(1,"II=1 continuous acceptance was not exercised");
        phase=0;$fclose(efd);$fclose(nfd);
        $display("PASS clear and II=1 bursts: exp=%0d norm=%0d",max_exp_run,max_norm_run);
        $display("PASS binary32 helpers, reciprocal and streaming stability");$finish;
    end
    initial begin #5000000;$fatal(1,"math timeout");end
endmodule
