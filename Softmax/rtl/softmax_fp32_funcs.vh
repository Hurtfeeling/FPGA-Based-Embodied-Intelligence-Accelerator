// Include inside each module using these functions. Deliberately no global guard.
// Portable synthesizable binary32 helpers. Inputs with exponent zero are DAZ;
// results use round-to-nearest-even followed by flush-to-zero (RNE/FTZ).
// No real-valued operators or vendor simulation primitives are used.
    function automatic fp32_is_nan(input [31:0] a);
        fp32_is_nan = (&a[30:23]) && (|a[22:0]);
    endfunction
    function automatic fp32_is_inf(input [31:0] a);
        fp32_is_inf = (&a[30:23]) && !(|a[22:0]);
    endfunction
    function automatic fp32_is_finite(input [31:0] a);
        fp32_is_finite = !(&a[30:23]);
    endfunction
    function automatic [31:0] fp32_flush(input [31:0] a);
        fp32_flush = (a[30:23] == 0) ? {a[31],31'b0} : a;
    endfunction
    function automatic fp32_lt(input [31:0] a, b);
        reg [31:0] aa,bb;
        begin
            aa=fp32_flush(a); bb=fp32_flush(b);
            if(fp32_is_nan(aa)||fp32_is_nan(bb)) fp32_lt=0;
            else if((aa[30:0]==0)&&(bb[30:0]==0)) fp32_lt=0;
            else if(aa[31]!=bb[31]) fp32_lt=aa[31];
            else if(aa[31]) fp32_lt=aa[30:0]>bb[30:0];
            else fp32_lt=aa[30:0]<bb[30:0];
        end
    endfunction
    function automatic [31:0] shr_jam32(input [31:0] a,input integer sh);
        reg lost;
        integer i;
        begin
            lost=0;
            for(i=0;i<32;i=i+1) if(i<sh) lost=lost|a[i];
            if(sh<=0) shr_jam32=a;
            else if(sh>=32) shr_jam32={31'b0,|a};
            else shr_jam32=(a>>sh)|{31'b0,lost};
        end
    endfunction
    function automatic [63:0] shr_jam64(input [63:0] a,input integer sh);
        reg lost;
        integer i;
        begin
            lost=0;
            for(i=0;i<64;i=i+1) if(i<sh) lost=lost|a[i];
            if(sh<=0) shr_jam64=a;
            else if(sh>=64) shr_jam64={63'b0,|a};
            else shr_jam64=(a>>sh)|{63'b0,lost};
        end
    endfunction
    // Binary-search leading-one detector: six mux levels rather than a chain
    // of conditional one-bit normalization shifts.
    function automatic integer msb_index64(input [63:0] a);
        reg [63:0] x;
        integer position;
        begin
            x=a;position=0;
            if(|x[63:32]) begin x=x>>32;position=position+32;end
            if(|x[31:16]) begin x=x>>16;position=position+16;end
            if(|x[15:8]) begin x=x>>8;position=position+8;end
            if(|x[7:4]) begin x=x>>4;position=position+4;end
            if(|x[3:2]) begin x=x>>2;position=position+2;end
            if(x[1]) position=position+1;
            msb_index64=position;
        end
    endfunction
    // sig[26:3] is the 24-bit significand, bits 2/1/0 are guard/round/sticky.
    function automatic [31:0] fp32_round_pack(
        input sign_bit,input integer exponent,input [31:0] sig);
        reg [31:0] s;
        reg [24:0] rounded;
        integer e;
        begin
            s=sig; e=exponent;
            if(e<=0) begin s=shr_jam32(s,1-e); e=1; end
            rounded={1'b0,s[26:3]} + (s[2] && (s[1]||s[0]||s[3]));
            if(rounded[24]) begin rounded=rounded>>1; e=e+1; end
            if(e>=255) fp32_round_pack={sign_bit,8'hff,23'b0};
            else if(!rounded[23]) fp32_round_pack={sign_bit,31'b0};
            else fp32_round_pack={sign_bit,e[7:0],rounded[22:0]};
        end
    endfunction
    function automatic [31:0] fp32_add(input [31:0] a,b);
        reg [31:0] aa,bb,hiword,loword,ls,ss,s;
        reg sign_bit;
        integer e,delta,lead;
        begin
            aa=fp32_flush(a); bb=fp32_flush(b);
            if(fp32_is_nan(aa)||fp32_is_nan(bb)) fp32_add=32'h7fc00000;
            else if(fp32_is_inf(aa)&&fp32_is_inf(bb)&&aa[31]!=bb[31]) fp32_add=32'h7fc00000;
            else if(fp32_is_inf(aa)) fp32_add=aa;
            else if(fp32_is_inf(bb)) fp32_add=bb;
            else if(aa[30:0]==0&&bb[30:0]==0) fp32_add={aa[31]&bb[31],31'b0};
            else if(aa[30:0]==0) fp32_add=bb;
            else if(bb[30:0]==0) fp32_add=aa;
            else begin
                if(aa[30:0]>=bb[30:0]) begin hiword=aa;loword=bb;end
                else begin hiword=bb;loword=aa;end
                sign_bit=hiword[31]; e={24'b0,hiword[30:23]};
                delta={24'b0,hiword[30:23]}-{24'b0,loword[30:23]};
                ls={5'b0,1'b1,hiword[22:0],3'b0};
                ss=shr_jam32({5'b0,1'b1,loword[22:0],3'b0},delta);
                if(hiword[31]==loword[31]) begin
                    s=ls+ss;
                    if(s[27]) begin s=shr_jam32(s,1);e=e+1;end
                end else begin
                    s=ls-ss;
                    lead=msb_index64({32'b0,s});
                    if(s!=0) begin s=s<<(26-lead);e=e-(26-lead);end
                end
                if(s==0) fp32_add=0;
                else fp32_add=fp32_round_pack(sign_bit,e,s);
            end
        end
    endfunction
    function automatic [31:0] fp32_sub(input [31:0] a,b);
        fp32_sub=fp32_add(a,{~b[31],b[30:0]});
    endfunction
    function automatic [31:0] fp32_mul(input [31:0] a,b);
        reg [31:0] aa,bb,s;
        reg [47:0] product;
        reg sign_bit;
        integer e;
        begin
            aa=fp32_flush(a);bb=fp32_flush(b);sign_bit=aa[31]^bb[31];
            if(fp32_is_nan(aa)||fp32_is_nan(bb)) fp32_mul=32'h7fc00000;
            else if((fp32_is_inf(aa)&&bb[30:0]==0)||(fp32_is_inf(bb)&&aa[30:0]==0)) fp32_mul=32'h7fc00000;
            else if(fp32_is_inf(aa)||fp32_is_inf(bb)) fp32_mul={sign_bit,8'hff,23'b0};
            else if(aa[30:0]==0||bb[30:0]==0) fp32_mul={sign_bit,31'b0};
            else begin
                product={1'b1,aa[22:0]}*{1'b1,bb[22:0]};
                e={24'b0,aa[30:23]}+{24'b0,bb[30:23]}-127;
                if(product[47]) begin s={5'b0,product[47:22],|product[21:0]};e=e+1;end
                else s={5'b0,product[46:21],|product[20:0]};
                fp32_mul=fp32_round_pack(sign_bit,e,s);
            end
        end
    endfunction
    function automatic [31:0] uq48_to_fp32(input [63:0] a);
        reg [63:0] aligned;
        integer lead;
        begin
            lead=msb_index64(a);
            if(lead>26) aligned=shr_jam64(a,lead-26);
            else aligned=a<<(26-lead);
            if(a==0) uq48_to_fp32=0;
            else uq48_to_fp32=fp32_round_pack(1'b0,lead-48+127,aligned[31:0]);
        end
    endfunction
    // Used only for nonnegative exp range reduction; caller clamps t>126.
    function automatic [31:0] fp32_to_uq23(input [31:0] a);
        reg [31:0] mant;
        integer shift;
        begin
            mant={8'b0,1'b1,a[22:0]};shift={24'b0,a[30:23]}-127;
            if(a[31]||a[30:23]==0) fp32_to_uq23=0;
            else if(shift>=32) fp32_to_uq23=32'hffffffff;
            else if(shift>=0) fp32_to_uq23=mant<<shift;
            else if(shift<=-32) fp32_to_uq23=0;
            else fp32_to_uq23=mant>>(-shift);
        end
    endfunction
    function automatic [31:0] uq30_scaled_to_fp32(input [31:0] a,input [6:0] n);
        reg [31:0] aligned;
        integer lead;
        begin
            lead=msb_index64({32'b0,a});
            if(lead>26) aligned=shr_jam32(a,lead-26);
            else aligned=a<<(26-lead);
            if(a==0) uq30_scaled_to_fp32=0;
            else uq30_scaled_to_fp32=fp32_round_pack(1'b0,lead-30-{25'b0,n}+127,aligned);
        end
    endfunction
