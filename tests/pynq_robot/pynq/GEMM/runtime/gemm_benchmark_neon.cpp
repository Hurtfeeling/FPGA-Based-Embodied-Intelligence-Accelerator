#include <omp.h>
#include <sched.h>
#include <time.h>
#include <arm_neon.h>
#include <cstdint>
#include <algorithm>

static inline __attribute__((always_inline)) void tile_0(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int bs, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s0_2=vdupq_n_s32(0);
  int32x4_t s0_3=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s1_2=vdupq_n_s32(0);
  int32x4_t s1_3=vdupq_n_s32(0);
  int32x4_t s2_0=vdupq_n_s32(0);
  int32x4_t s2_1=vdupq_n_s32(0);
  int32x4_t s2_2=vdupq_n_s32(0);
  int32x4_t s2_3=vdupq_n_s32(0);
  int32x4_t s3_0=vdupq_n_s32(0);
  int32x4_t s3_1=vdupq_n_s32(0);
  int32x4_t s3_2=vdupq_n_s32(0);
  int32x4_t s3_3=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t a2=vld1q_s8(a+(i+2)*k+p);
    const int8x16_t a3=vld1q_s8(a+(i+3)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*bs+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*bs+p);
    const int8x16_t b2=vld1q_s8(bt+(j+2)*bs+p);
    const int8x16_t b3=vld1q_s8(bt+(j+3)*bs+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s0_2=vpadalq_s16(s0_2,vmull_s8(vget_low_s8(a0),vget_low_s8(b2)));
    s0_2=vpadalq_s16(s0_2,vmull_high_s8(a0,b2));
    s0_3=vpadalq_s16(s0_3,vmull_s8(vget_low_s8(a0),vget_low_s8(b3)));
    s0_3=vpadalq_s16(s0_3,vmull_high_s8(a0,b3));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s1_2=vpadalq_s16(s1_2,vmull_s8(vget_low_s8(a1),vget_low_s8(b2)));
    s1_2=vpadalq_s16(s1_2,vmull_high_s8(a1,b2));
    s1_3=vpadalq_s16(s1_3,vmull_s8(vget_low_s8(a1),vget_low_s8(b3)));
    s1_3=vpadalq_s16(s1_3,vmull_high_s8(a1,b3));
    s2_0=vpadalq_s16(s2_0,vmull_s8(vget_low_s8(a2),vget_low_s8(b0)));
    s2_0=vpadalq_s16(s2_0,vmull_high_s8(a2,b0));
    s2_1=vpadalq_s16(s2_1,vmull_s8(vget_low_s8(a2),vget_low_s8(b1)));
    s2_1=vpadalq_s16(s2_1,vmull_high_s8(a2,b1));
    s2_2=vpadalq_s16(s2_2,vmull_s8(vget_low_s8(a2),vget_low_s8(b2)));
    s2_2=vpadalq_s16(s2_2,vmull_high_s8(a2,b2));
    s2_3=vpadalq_s16(s2_3,vmull_s8(vget_low_s8(a2),vget_low_s8(b3)));
    s2_3=vpadalq_s16(s2_3,vmull_high_s8(a2,b3));
    s3_0=vpadalq_s16(s3_0,vmull_s8(vget_low_s8(a3),vget_low_s8(b0)));
    s3_0=vpadalq_s16(s3_0,vmull_high_s8(a3,b0));
    s3_1=vpadalq_s16(s3_1,vmull_s8(vget_low_s8(a3),vget_low_s8(b1)));
    s3_1=vpadalq_s16(s3_1,vmull_high_s8(a3,b1));
    s3_2=vpadalq_s16(s3_2,vmull_s8(vget_low_s8(a3),vget_low_s8(b2)));
    s3_2=vpadalq_s16(s3_2,vmull_high_s8(a3,b2));
    s3_3=vpadalq_s16(s3_3,vmull_s8(vget_low_s8(a3),vget_low_s8(b3)));
    s3_3=vpadalq_s16(s3_3,vmull_high_s8(a3,b3));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+0)*n+j+2]=vaddvq_s32(s0_2);
  c[(i+0)*n+j+3]=vaddvq_s32(s0_3);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+1)*n+j+2]=vaddvq_s32(s1_2);
  c[(i+1)*n+j+3]=vaddvq_s32(s1_3);
  c[(i+2)*n+j+0]=vaddvq_s32(s2_0);
  c[(i+2)*n+j+1]=vaddvq_s32(s2_1);
  c[(i+2)*n+j+2]=vaddvq_s32(s2_2);
  c[(i+2)*n+j+3]=vaddvq_s32(s2_3);
  c[(i+3)*n+j+0]=vaddvq_s32(s3_0);
  c[(i+3)*n+j+1]=vaddvq_s32(s3_1);
  c[(i+3)*n+j+2]=vaddvq_s32(s3_2);
  c[(i+3)*n+j+3]=vaddvq_s32(s3_3);
}

static inline __attribute__((always_inline)) void tile_1(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int bs, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s0_2=vdupq_n_s32(0);
  int32x4_t s0_3=vdupq_n_s32(0);
  int32x4_t s0_4=vdupq_n_s32(0);
  int32x4_t s0_5=vdupq_n_s32(0);
  int32x4_t s0_6=vdupq_n_s32(0);
  int32x4_t s0_7=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s1_2=vdupq_n_s32(0);
  int32x4_t s1_3=vdupq_n_s32(0);
  int32x4_t s1_4=vdupq_n_s32(0);
  int32x4_t s1_5=vdupq_n_s32(0);
  int32x4_t s1_6=vdupq_n_s32(0);
  int32x4_t s1_7=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*bs+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*bs+p);
    const int8x16_t b2=vld1q_s8(bt+(j+2)*bs+p);
    const int8x16_t b3=vld1q_s8(bt+(j+3)*bs+p);
    const int8x16_t b4=vld1q_s8(bt+(j+4)*bs+p);
    const int8x16_t b5=vld1q_s8(bt+(j+5)*bs+p);
    const int8x16_t b6=vld1q_s8(bt+(j+6)*bs+p);
    const int8x16_t b7=vld1q_s8(bt+(j+7)*bs+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s0_2=vpadalq_s16(s0_2,vmull_s8(vget_low_s8(a0),vget_low_s8(b2)));
    s0_2=vpadalq_s16(s0_2,vmull_high_s8(a0,b2));
    s0_3=vpadalq_s16(s0_3,vmull_s8(vget_low_s8(a0),vget_low_s8(b3)));
    s0_3=vpadalq_s16(s0_3,vmull_high_s8(a0,b3));
    s0_4=vpadalq_s16(s0_4,vmull_s8(vget_low_s8(a0),vget_low_s8(b4)));
    s0_4=vpadalq_s16(s0_4,vmull_high_s8(a0,b4));
    s0_5=vpadalq_s16(s0_5,vmull_s8(vget_low_s8(a0),vget_low_s8(b5)));
    s0_5=vpadalq_s16(s0_5,vmull_high_s8(a0,b5));
    s0_6=vpadalq_s16(s0_6,vmull_s8(vget_low_s8(a0),vget_low_s8(b6)));
    s0_6=vpadalq_s16(s0_6,vmull_high_s8(a0,b6));
    s0_7=vpadalq_s16(s0_7,vmull_s8(vget_low_s8(a0),vget_low_s8(b7)));
    s0_7=vpadalq_s16(s0_7,vmull_high_s8(a0,b7));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s1_2=vpadalq_s16(s1_2,vmull_s8(vget_low_s8(a1),vget_low_s8(b2)));
    s1_2=vpadalq_s16(s1_2,vmull_high_s8(a1,b2));
    s1_3=vpadalq_s16(s1_3,vmull_s8(vget_low_s8(a1),vget_low_s8(b3)));
    s1_3=vpadalq_s16(s1_3,vmull_high_s8(a1,b3));
    s1_4=vpadalq_s16(s1_4,vmull_s8(vget_low_s8(a1),vget_low_s8(b4)));
    s1_4=vpadalq_s16(s1_4,vmull_high_s8(a1,b4));
    s1_5=vpadalq_s16(s1_5,vmull_s8(vget_low_s8(a1),vget_low_s8(b5)));
    s1_5=vpadalq_s16(s1_5,vmull_high_s8(a1,b5));
    s1_6=vpadalq_s16(s1_6,vmull_s8(vget_low_s8(a1),vget_low_s8(b6)));
    s1_6=vpadalq_s16(s1_6,vmull_high_s8(a1,b6));
    s1_7=vpadalq_s16(s1_7,vmull_s8(vget_low_s8(a1),vget_low_s8(b7)));
    s1_7=vpadalq_s16(s1_7,vmull_high_s8(a1,b7));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+0)*n+j+2]=vaddvq_s32(s0_2);
  c[(i+0)*n+j+3]=vaddvq_s32(s0_3);
  c[(i+0)*n+j+4]=vaddvq_s32(s0_4);
  c[(i+0)*n+j+5]=vaddvq_s32(s0_5);
  c[(i+0)*n+j+6]=vaddvq_s32(s0_6);
  c[(i+0)*n+j+7]=vaddvq_s32(s0_7);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+1)*n+j+2]=vaddvq_s32(s1_2);
  c[(i+1)*n+j+3]=vaddvq_s32(s1_3);
  c[(i+1)*n+j+4]=vaddvq_s32(s1_4);
  c[(i+1)*n+j+5]=vaddvq_s32(s1_5);
  c[(i+1)*n+j+6]=vaddvq_s32(s1_6);
  c[(i+1)*n+j+7]=vaddvq_s32(s1_7);
}

static inline __attribute__((always_inline)) void tile_2(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int bs, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s2_0=vdupq_n_s32(0);
  int32x4_t s2_1=vdupq_n_s32(0);
  int32x4_t s3_0=vdupq_n_s32(0);
  int32x4_t s3_1=vdupq_n_s32(0);
  int32x4_t s4_0=vdupq_n_s32(0);
  int32x4_t s4_1=vdupq_n_s32(0);
  int32x4_t s5_0=vdupq_n_s32(0);
  int32x4_t s5_1=vdupq_n_s32(0);
  int32x4_t s6_0=vdupq_n_s32(0);
  int32x4_t s6_1=vdupq_n_s32(0);
  int32x4_t s7_0=vdupq_n_s32(0);
  int32x4_t s7_1=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t a2=vld1q_s8(a+(i+2)*k+p);
    const int8x16_t a3=vld1q_s8(a+(i+3)*k+p);
    const int8x16_t a4=vld1q_s8(a+(i+4)*k+p);
    const int8x16_t a5=vld1q_s8(a+(i+5)*k+p);
    const int8x16_t a6=vld1q_s8(a+(i+6)*k+p);
    const int8x16_t a7=vld1q_s8(a+(i+7)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*bs+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*bs+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s2_0=vpadalq_s16(s2_0,vmull_s8(vget_low_s8(a2),vget_low_s8(b0)));
    s2_0=vpadalq_s16(s2_0,vmull_high_s8(a2,b0));
    s2_1=vpadalq_s16(s2_1,vmull_s8(vget_low_s8(a2),vget_low_s8(b1)));
    s2_1=vpadalq_s16(s2_1,vmull_high_s8(a2,b1));
    s3_0=vpadalq_s16(s3_0,vmull_s8(vget_low_s8(a3),vget_low_s8(b0)));
    s3_0=vpadalq_s16(s3_0,vmull_high_s8(a3,b0));
    s3_1=vpadalq_s16(s3_1,vmull_s8(vget_low_s8(a3),vget_low_s8(b1)));
    s3_1=vpadalq_s16(s3_1,vmull_high_s8(a3,b1));
    s4_0=vpadalq_s16(s4_0,vmull_s8(vget_low_s8(a4),vget_low_s8(b0)));
    s4_0=vpadalq_s16(s4_0,vmull_high_s8(a4,b0));
    s4_1=vpadalq_s16(s4_1,vmull_s8(vget_low_s8(a4),vget_low_s8(b1)));
    s4_1=vpadalq_s16(s4_1,vmull_high_s8(a4,b1));
    s5_0=vpadalq_s16(s5_0,vmull_s8(vget_low_s8(a5),vget_low_s8(b0)));
    s5_0=vpadalq_s16(s5_0,vmull_high_s8(a5,b0));
    s5_1=vpadalq_s16(s5_1,vmull_s8(vget_low_s8(a5),vget_low_s8(b1)));
    s5_1=vpadalq_s16(s5_1,vmull_high_s8(a5,b1));
    s6_0=vpadalq_s16(s6_0,vmull_s8(vget_low_s8(a6),vget_low_s8(b0)));
    s6_0=vpadalq_s16(s6_0,vmull_high_s8(a6,b0));
    s6_1=vpadalq_s16(s6_1,vmull_s8(vget_low_s8(a6),vget_low_s8(b1)));
    s6_1=vpadalq_s16(s6_1,vmull_high_s8(a6,b1));
    s7_0=vpadalq_s16(s7_0,vmull_s8(vget_low_s8(a7),vget_low_s8(b0)));
    s7_0=vpadalq_s16(s7_0,vmull_high_s8(a7,b0));
    s7_1=vpadalq_s16(s7_1,vmull_s8(vget_low_s8(a7),vget_low_s8(b1)));
    s7_1=vpadalq_s16(s7_1,vmull_high_s8(a7,b1));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+2)*n+j+0]=vaddvq_s32(s2_0);
  c[(i+2)*n+j+1]=vaddvq_s32(s2_1);
  c[(i+3)*n+j+0]=vaddvq_s32(s3_0);
  c[(i+3)*n+j+1]=vaddvq_s32(s3_1);
  c[(i+4)*n+j+0]=vaddvq_s32(s4_0);
  c[(i+4)*n+j+1]=vaddvq_s32(s4_1);
  c[(i+5)*n+j+0]=vaddvq_s32(s5_0);
  c[(i+5)*n+j+1]=vaddvq_s32(s5_1);
  c[(i+6)*n+j+0]=vaddvq_s32(s6_0);
  c[(i+6)*n+j+1]=vaddvq_s32(s6_1);
  c[(i+7)*n+j+0]=vaddvq_s32(s7_0);
  c[(i+7)*n+j+1]=vaddvq_s32(s7_1);
}


static inline void transpose8(const int8_t* b,int8_t* bt,int n,int bs){
  int8x8_t r[8];
  for(int i=0;i<8;++i) r[i]=vld1_s8(b+i*n);
  auto t0=vtrn_s8(r[0],r[1]); auto t1=vtrn_s8(r[2],r[3]);
  auto t2=vtrn_s8(r[4],r[5]); auto t3=vtrn_s8(r[6],r[7]);
  auto u0=vtrn_s16(vreinterpret_s16_s8(t0.val[0]),vreinterpret_s16_s8(t1.val[0]));
  auto u1=vtrn_s16(vreinterpret_s16_s8(t0.val[1]),vreinterpret_s16_s8(t1.val[1]));
  auto u2=vtrn_s16(vreinterpret_s16_s8(t2.val[0]),vreinterpret_s16_s8(t3.val[0]));
  auto u3=vtrn_s16(vreinterpret_s16_s8(t2.val[1]),vreinterpret_s16_s8(t3.val[1]));
  auto v0=vtrn_s32(vreinterpret_s32_s16(u0.val[0]),vreinterpret_s32_s16(u2.val[0]));
  auto v1=vtrn_s32(vreinterpret_s32_s16(u1.val[0]),vreinterpret_s32_s16(u3.val[0]));
  auto v2=vtrn_s32(vreinterpret_s32_s16(u0.val[1]),vreinterpret_s32_s16(u2.val[1]));
  auto v3=vtrn_s32(vreinterpret_s32_s16(u1.val[1]),vreinterpret_s32_s16(u3.val[1]));
  vst1_s8(bt+0*bs,vreinterpret_s8_s32(v0.val[0]));
  vst1_s8(bt+1*bs,vreinterpret_s8_s32(v1.val[0]));
  vst1_s8(bt+2*bs,vreinterpret_s8_s32(v2.val[0]));
  vst1_s8(bt+3*bs,vreinterpret_s8_s32(v3.val[0]));
  vst1_s8(bt+4*bs,vreinterpret_s8_s32(v0.val[1]));
  vst1_s8(bt+5*bs,vreinterpret_s8_s32(v1.val[1]));
  vst1_s8(bt+6*bs,vreinterpret_s8_s32(v2.val[1]));
  vst1_s8(bt+7*bs,vreinterpret_s8_s32(v3.val[1]));
}
extern "C" int fast_pack(const int8_t* b,int8_t* bt,int k,int n,int bs,int first,int last,int mode,int block){
  if(!b||!bt||k<=0||n<=0||k%8||n%8||bs<k||bs%8||first<0||last>n||first>=last||first%8||last%8||mode<0||mode>1||block<8||block%8) return -1;
  for(int jj=first;jj<last;jj+=block)
    for(int kk=0;kk<k;kk+=block){
      int je=std::min(jj+block,last),ke=std::min(kk+block,k);
      if(mode==0){
        for(int p=kk;p<ke;++p) for(int q=jj;q<je;++q) bt[q*bs+p]=b[p*n+q];
      }else{
        for(int p=kk;p<ke;p+=8) for(int q=jj;q<je;q+=8) transpose8(b+p*n+q,bt+q*bs+p,n,bs);
      }
    }
  return 0;
}
static inline __attribute__((always_inline)) void packed_tile_0(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s0_2=vdupq_n_s32(0);
  int32x4_t s0_3=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s1_2=vdupq_n_s32(0);
  int32x4_t s1_3=vdupq_n_s32(0);
  int32x4_t s2_0=vdupq_n_s32(0);
  int32x4_t s2_1=vdupq_n_s32(0);
  int32x4_t s2_2=vdupq_n_s32(0);
  int32x4_t s2_3=vdupq_n_s32(0);
  int32x4_t s3_0=vdupq_n_s32(0);
  int32x4_t s3_1=vdupq_n_s32(0);
  int32x4_t s3_2=vdupq_n_s32(0);
  int32x4_t s3_3=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t a2=vld1q_s8(a+(i+2)*k+p);
    const int8x16_t a3=vld1q_s8(a+(i+3)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*k+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*k+p);
    const int8x16_t b2=vld1q_s8(bt+(j+2)*k+p);
    const int8x16_t b3=vld1q_s8(bt+(j+3)*k+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s0_2=vpadalq_s16(s0_2,vmull_s8(vget_low_s8(a0),vget_low_s8(b2)));
    s0_2=vpadalq_s16(s0_2,vmull_high_s8(a0,b2));
    s0_3=vpadalq_s16(s0_3,vmull_s8(vget_low_s8(a0),vget_low_s8(b3)));
    s0_3=vpadalq_s16(s0_3,vmull_high_s8(a0,b3));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s1_2=vpadalq_s16(s1_2,vmull_s8(vget_low_s8(a1),vget_low_s8(b2)));
    s1_2=vpadalq_s16(s1_2,vmull_high_s8(a1,b2));
    s1_3=vpadalq_s16(s1_3,vmull_s8(vget_low_s8(a1),vget_low_s8(b3)));
    s1_3=vpadalq_s16(s1_3,vmull_high_s8(a1,b3));
    s2_0=vpadalq_s16(s2_0,vmull_s8(vget_low_s8(a2),vget_low_s8(b0)));
    s2_0=vpadalq_s16(s2_0,vmull_high_s8(a2,b0));
    s2_1=vpadalq_s16(s2_1,vmull_s8(vget_low_s8(a2),vget_low_s8(b1)));
    s2_1=vpadalq_s16(s2_1,vmull_high_s8(a2,b1));
    s2_2=vpadalq_s16(s2_2,vmull_s8(vget_low_s8(a2),vget_low_s8(b2)));
    s2_2=vpadalq_s16(s2_2,vmull_high_s8(a2,b2));
    s2_3=vpadalq_s16(s2_3,vmull_s8(vget_low_s8(a2),vget_low_s8(b3)));
    s2_3=vpadalq_s16(s2_3,vmull_high_s8(a2,b3));
    s3_0=vpadalq_s16(s3_0,vmull_s8(vget_low_s8(a3),vget_low_s8(b0)));
    s3_0=vpadalq_s16(s3_0,vmull_high_s8(a3,b0));
    s3_1=vpadalq_s16(s3_1,vmull_s8(vget_low_s8(a3),vget_low_s8(b1)));
    s3_1=vpadalq_s16(s3_1,vmull_high_s8(a3,b1));
    s3_2=vpadalq_s16(s3_2,vmull_s8(vget_low_s8(a3),vget_low_s8(b2)));
    s3_2=vpadalq_s16(s3_2,vmull_high_s8(a3,b2));
    s3_3=vpadalq_s16(s3_3,vmull_s8(vget_low_s8(a3),vget_low_s8(b3)));
    s3_3=vpadalq_s16(s3_3,vmull_high_s8(a3,b3));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+0)*n+j+2]=vaddvq_s32(s0_2);
  c[(i+0)*n+j+3]=vaddvq_s32(s0_3);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+1)*n+j+2]=vaddvq_s32(s1_2);
  c[(i+1)*n+j+3]=vaddvq_s32(s1_3);
  c[(i+2)*n+j+0]=vaddvq_s32(s2_0);
  c[(i+2)*n+j+1]=vaddvq_s32(s2_1);
  c[(i+2)*n+j+2]=vaddvq_s32(s2_2);
  c[(i+2)*n+j+3]=vaddvq_s32(s2_3);
  c[(i+3)*n+j+0]=vaddvq_s32(s3_0);
  c[(i+3)*n+j+1]=vaddvq_s32(s3_1);
  c[(i+3)*n+j+2]=vaddvq_s32(s3_2);
  c[(i+3)*n+j+3]=vaddvq_s32(s3_3);
}

static inline __attribute__((always_inline)) void packed_tile_1(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s0_2=vdupq_n_s32(0);
  int32x4_t s0_3=vdupq_n_s32(0);
  int32x4_t s0_4=vdupq_n_s32(0);
  int32x4_t s0_5=vdupq_n_s32(0);
  int32x4_t s0_6=vdupq_n_s32(0);
  int32x4_t s0_7=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s1_2=vdupq_n_s32(0);
  int32x4_t s1_3=vdupq_n_s32(0);
  int32x4_t s1_4=vdupq_n_s32(0);
  int32x4_t s1_5=vdupq_n_s32(0);
  int32x4_t s1_6=vdupq_n_s32(0);
  int32x4_t s1_7=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*k+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*k+p);
    const int8x16_t b2=vld1q_s8(bt+(j+2)*k+p);
    const int8x16_t b3=vld1q_s8(bt+(j+3)*k+p);
    const int8x16_t b4=vld1q_s8(bt+(j+4)*k+p);
    const int8x16_t b5=vld1q_s8(bt+(j+5)*k+p);
    const int8x16_t b6=vld1q_s8(bt+(j+6)*k+p);
    const int8x16_t b7=vld1q_s8(bt+(j+7)*k+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s0_2=vpadalq_s16(s0_2,vmull_s8(vget_low_s8(a0),vget_low_s8(b2)));
    s0_2=vpadalq_s16(s0_2,vmull_high_s8(a0,b2));
    s0_3=vpadalq_s16(s0_3,vmull_s8(vget_low_s8(a0),vget_low_s8(b3)));
    s0_3=vpadalq_s16(s0_3,vmull_high_s8(a0,b3));
    s0_4=vpadalq_s16(s0_4,vmull_s8(vget_low_s8(a0),vget_low_s8(b4)));
    s0_4=vpadalq_s16(s0_4,vmull_high_s8(a0,b4));
    s0_5=vpadalq_s16(s0_5,vmull_s8(vget_low_s8(a0),vget_low_s8(b5)));
    s0_5=vpadalq_s16(s0_5,vmull_high_s8(a0,b5));
    s0_6=vpadalq_s16(s0_6,vmull_s8(vget_low_s8(a0),vget_low_s8(b6)));
    s0_6=vpadalq_s16(s0_6,vmull_high_s8(a0,b6));
    s0_7=vpadalq_s16(s0_7,vmull_s8(vget_low_s8(a0),vget_low_s8(b7)));
    s0_7=vpadalq_s16(s0_7,vmull_high_s8(a0,b7));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s1_2=vpadalq_s16(s1_2,vmull_s8(vget_low_s8(a1),vget_low_s8(b2)));
    s1_2=vpadalq_s16(s1_2,vmull_high_s8(a1,b2));
    s1_3=vpadalq_s16(s1_3,vmull_s8(vget_low_s8(a1),vget_low_s8(b3)));
    s1_3=vpadalq_s16(s1_3,vmull_high_s8(a1,b3));
    s1_4=vpadalq_s16(s1_4,vmull_s8(vget_low_s8(a1),vget_low_s8(b4)));
    s1_4=vpadalq_s16(s1_4,vmull_high_s8(a1,b4));
    s1_5=vpadalq_s16(s1_5,vmull_s8(vget_low_s8(a1),vget_low_s8(b5)));
    s1_5=vpadalq_s16(s1_5,vmull_high_s8(a1,b5));
    s1_6=vpadalq_s16(s1_6,vmull_s8(vget_low_s8(a1),vget_low_s8(b6)));
    s1_6=vpadalq_s16(s1_6,vmull_high_s8(a1,b6));
    s1_7=vpadalq_s16(s1_7,vmull_s8(vget_low_s8(a1),vget_low_s8(b7)));
    s1_7=vpadalq_s16(s1_7,vmull_high_s8(a1,b7));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+0)*n+j+2]=vaddvq_s32(s0_2);
  c[(i+0)*n+j+3]=vaddvq_s32(s0_3);
  c[(i+0)*n+j+4]=vaddvq_s32(s0_4);
  c[(i+0)*n+j+5]=vaddvq_s32(s0_5);
  c[(i+0)*n+j+6]=vaddvq_s32(s0_6);
  c[(i+0)*n+j+7]=vaddvq_s32(s0_7);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+1)*n+j+2]=vaddvq_s32(s1_2);
  c[(i+1)*n+j+3]=vaddvq_s32(s1_3);
  c[(i+1)*n+j+4]=vaddvq_s32(s1_4);
  c[(i+1)*n+j+5]=vaddvq_s32(s1_5);
  c[(i+1)*n+j+6]=vaddvq_s32(s1_6);
  c[(i+1)*n+j+7]=vaddvq_s32(s1_7);
}

static inline __attribute__((always_inline)) void packed_tile_2(const int8_t* __restrict a, const int8_t* __restrict bt, int32_t* __restrict c, int n, int k, int i, int j) {
  int32x4_t s0_0=vdupq_n_s32(0);
  int32x4_t s0_1=vdupq_n_s32(0);
  int32x4_t s1_0=vdupq_n_s32(0);
  int32x4_t s1_1=vdupq_n_s32(0);
  int32x4_t s2_0=vdupq_n_s32(0);
  int32x4_t s2_1=vdupq_n_s32(0);
  int32x4_t s3_0=vdupq_n_s32(0);
  int32x4_t s3_1=vdupq_n_s32(0);
  int32x4_t s4_0=vdupq_n_s32(0);
  int32x4_t s4_1=vdupq_n_s32(0);
  int32x4_t s5_0=vdupq_n_s32(0);
  int32x4_t s5_1=vdupq_n_s32(0);
  int32x4_t s6_0=vdupq_n_s32(0);
  int32x4_t s6_1=vdupq_n_s32(0);
  int32x4_t s7_0=vdupq_n_s32(0);
  int32x4_t s7_1=vdupq_n_s32(0);
  for(int p=0;p<k;p+=16){
    const int8x16_t a0=vld1q_s8(a+(i+0)*k+p);
    const int8x16_t a1=vld1q_s8(a+(i+1)*k+p);
    const int8x16_t a2=vld1q_s8(a+(i+2)*k+p);
    const int8x16_t a3=vld1q_s8(a+(i+3)*k+p);
    const int8x16_t a4=vld1q_s8(a+(i+4)*k+p);
    const int8x16_t a5=vld1q_s8(a+(i+5)*k+p);
    const int8x16_t a6=vld1q_s8(a+(i+6)*k+p);
    const int8x16_t a7=vld1q_s8(a+(i+7)*k+p);
    const int8x16_t b0=vld1q_s8(bt+(j+0)*k+p);
    const int8x16_t b1=vld1q_s8(bt+(j+1)*k+p);
    s0_0=vpadalq_s16(s0_0,vmull_s8(vget_low_s8(a0),vget_low_s8(b0)));
    s0_0=vpadalq_s16(s0_0,vmull_high_s8(a0,b0));
    s0_1=vpadalq_s16(s0_1,vmull_s8(vget_low_s8(a0),vget_low_s8(b1)));
    s0_1=vpadalq_s16(s0_1,vmull_high_s8(a0,b1));
    s1_0=vpadalq_s16(s1_0,vmull_s8(vget_low_s8(a1),vget_low_s8(b0)));
    s1_0=vpadalq_s16(s1_0,vmull_high_s8(a1,b0));
    s1_1=vpadalq_s16(s1_1,vmull_s8(vget_low_s8(a1),vget_low_s8(b1)));
    s1_1=vpadalq_s16(s1_1,vmull_high_s8(a1,b1));
    s2_0=vpadalq_s16(s2_0,vmull_s8(vget_low_s8(a2),vget_low_s8(b0)));
    s2_0=vpadalq_s16(s2_0,vmull_high_s8(a2,b0));
    s2_1=vpadalq_s16(s2_1,vmull_s8(vget_low_s8(a2),vget_low_s8(b1)));
    s2_1=vpadalq_s16(s2_1,vmull_high_s8(a2,b1));
    s3_0=vpadalq_s16(s3_0,vmull_s8(vget_low_s8(a3),vget_low_s8(b0)));
    s3_0=vpadalq_s16(s3_0,vmull_high_s8(a3,b0));
    s3_1=vpadalq_s16(s3_1,vmull_s8(vget_low_s8(a3),vget_low_s8(b1)));
    s3_1=vpadalq_s16(s3_1,vmull_high_s8(a3,b1));
    s4_0=vpadalq_s16(s4_0,vmull_s8(vget_low_s8(a4),vget_low_s8(b0)));
    s4_0=vpadalq_s16(s4_0,vmull_high_s8(a4,b0));
    s4_1=vpadalq_s16(s4_1,vmull_s8(vget_low_s8(a4),vget_low_s8(b1)));
    s4_1=vpadalq_s16(s4_1,vmull_high_s8(a4,b1));
    s5_0=vpadalq_s16(s5_0,vmull_s8(vget_low_s8(a5),vget_low_s8(b0)));
    s5_0=vpadalq_s16(s5_0,vmull_high_s8(a5,b0));
    s5_1=vpadalq_s16(s5_1,vmull_s8(vget_low_s8(a5),vget_low_s8(b1)));
    s5_1=vpadalq_s16(s5_1,vmull_high_s8(a5,b1));
    s6_0=vpadalq_s16(s6_0,vmull_s8(vget_low_s8(a6),vget_low_s8(b0)));
    s6_0=vpadalq_s16(s6_0,vmull_high_s8(a6,b0));
    s6_1=vpadalq_s16(s6_1,vmull_s8(vget_low_s8(a6),vget_low_s8(b1)));
    s6_1=vpadalq_s16(s6_1,vmull_high_s8(a6,b1));
    s7_0=vpadalq_s16(s7_0,vmull_s8(vget_low_s8(a7),vget_low_s8(b0)));
    s7_0=vpadalq_s16(s7_0,vmull_high_s8(a7,b0));
    s7_1=vpadalq_s16(s7_1,vmull_s8(vget_low_s8(a7),vget_low_s8(b1)));
    s7_1=vpadalq_s16(s7_1,vmull_high_s8(a7,b1));
  }
  c[(i+0)*n+j+0]=vaddvq_s32(s0_0);
  c[(i+0)*n+j+1]=vaddvq_s32(s0_1);
  c[(i+1)*n+j+0]=vaddvq_s32(s1_0);
  c[(i+1)*n+j+1]=vaddvq_s32(s1_1);
  c[(i+2)*n+j+0]=vaddvq_s32(s2_0);
  c[(i+2)*n+j+1]=vaddvq_s32(s2_1);
  c[(i+3)*n+j+0]=vaddvq_s32(s3_0);
  c[(i+3)*n+j+1]=vaddvq_s32(s3_1);
  c[(i+4)*n+j+0]=vaddvq_s32(s4_0);
  c[(i+4)*n+j+1]=vaddvq_s32(s4_1);
  c[(i+5)*n+j+0]=vaddvq_s32(s5_0);
  c[(i+5)*n+j+1]=vaddvq_s32(s5_1);
  c[(i+6)*n+j+0]=vaddvq_s32(s6_0);
  c[(i+6)*n+j+1]=vaddvq_s32(s6_1);
  c[(i+7)*n+j+0]=vaddvq_s32(s7_0);
  c[(i+7)*n+j+1]=vaddvq_s32(s7_1);
}


extern "C" int fast_gemm(const int8_t* a,const int8_t* bt,int32_t* c,
 int m,int n,int k,int bs,int first,int last,int variant,int mb,int nb){
  if(!a||!bt||!c||m<=0||n<=0||k<=0||k>16352||m%32||n%32||k%32||bs<k||
     first<0||last>m||first>=last||first%8||last%8||mb<=0||nb<=0||mb%8||nb%8||variant<0||variant>2) return -1;
  if(bs==k){
    for(int ii=first;ii<last;ii+=mb){
      int imax=std::min(ii+mb,last);
      for(int jj=0;jj<n;jj+=nb){
        int jmax=std::min(jj+nb,n);
        if(variant==0){for(int i=ii;i<imax;i+=4)for(int j=jj;j<jmax;j+=4)packed_tile_0(a,bt,c,n,k,i,j);}
        if(variant==1){for(int i=ii;i<imax;i+=2)for(int j=jj;j<jmax;j+=8)packed_tile_1(a,bt,c,n,k,i,j);}
        if(variant==2){for(int i=ii;i<imax;i+=8)for(int j=jj;j<jmax;j+=2)packed_tile_2(a,bt,c,n,k,i,j);}
      }
    }
    return 0;
  }
  for(int ii=first;ii<last;ii+=mb){
    int imax=std::min(ii+mb,last);
    for(int jj=0;jj<n;jj+=nb){
      int jmax=std::min(jj+nb,n);
      if(variant==0){for(int i=ii;i<imax;i+=4)for(int j=jj;j<jmax;j+=4)tile_0(a,bt,c,n,k,bs,i,j);}
      if(variant==1){for(int i=ii;i<imax;i+=2)for(int j=jj;j<jmax;j+=8)tile_1(a,bt,c,n,k,bs,i,j);}
      if(variant==2){for(int i=ii;i<imax;i+=8)for(int j=jj;j<jmax;j+=2)tile_2(a,bt,c,n,k,bs,i,j);}
    }
  }
  return 0;
}

static double wall_time(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double cpu_time(){struct timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int bind_core(int core,cpu_set_t* previous){
 if(core<0||core>=CPU_SETSIZE||sched_getaffinity(0,sizeof(*previous),previous))return -1;
 cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(core,&mask);
 return sched_setaffinity(0,sizeof(mask),&mask);
}
extern "C" int fast_pack_parallel(const int8_t* b,int8_t* bt,int k,int n,int bs,int mode,int block,int threads,const int* ids,double* metrics){
 if(!ids||!metrics||(threads!=1&&threads!=4))return -1;
 double begin=wall_time(),pack_begin=begin,ends[4]={0,0,0,0};
 int error=0,old_dynamic=omp_get_dynamic();omp_set_dynamic(0);
 #pragma omp parallel num_threads(threads) reduction(max:error) shared(pack_begin,ends)
 {
  int index=omp_get_thread_num(),parts=omp_get_num_threads();
  cpu_set_t previous;
  int bound=bind_core(ids[index],&previous);
  if(bound||parts!=threads)error=1;
  #pragma omp barrier
  #pragma omp single
  {pack_begin=wall_time();}
  int first=index*(n/8)/parts*8,last=(index+1)*(n/8)/parts*8;
  if(first<last&&fast_pack(b,bt,k,n,bs,first,last,mode,block))error=1;
  ends[index]=wall_time();
  if(!bound&&sched_setaffinity(0,sizeof(previous),&previous))error=1;
 }
 double end=wall_time(),pack_end=*std::max_element(ends,ends+threads);
 metrics[0]=pack_end-pack_begin;metrics[1]=pack_begin-begin;metrics[2]=end-pack_end;
 omp_set_dynamic(old_dynamic);
 return error?-1:0;
}
extern "C" int fast_cpu_parallel(const int8_t* a,const int8_t* bt_const,int32_t* c,
 int m,int n,int k,int bs,int variant,int mb,int nb,int mode,int block,int pack_threads,const int* ids,double* metrics,const int8_t* b){
 if(!ids||!metrics||!b||(pack_threads!=1&&pack_threads!=4))return -1;
 int8_t* bt=const_cast<int8_t*>(bt_const);
 int error=0,old_dynamic=omp_get_dynamic();omp_set_dynamic(0);
 double begin=wall_time(),compute_begin=begin,pack_begin=begin,cpu[4]={0,0,0,0},ends[4]={0,0,0,0};
 #pragma omp parallel num_threads(4) reduction(max:error) shared(compute_begin,pack_begin,cpu,ends)
 {
  int index=omp_get_thread_num(),parts=omp_get_num_threads();
  cpu_set_t previous;
  int bound=bind_core(ids[index],&previous);
  if(bound||parts!=4)error=1;
  #pragma omp barrier
  #pragma omp single
  {pack_begin=wall_time();}
  if(index<pack_threads){
   int first=index*(n/8)/pack_threads*8,last=(index+1)*(n/8)/pack_threads*8;
   if(first<last&&fast_pack(b,bt,k,n,bs,first,last,mode,block))error=1;
  }
  #pragma omp barrier
  #pragma omp single
  {compute_begin=wall_time();}
  double t=cpu_time();
  int first=index*(m/8)/parts*8,last=(index+1)*(m/8)/parts*8;
  if(fast_gemm(a,bt,c,m,n,k,bs,first,last,variant,mb,nb))error=1;
  cpu[index]=cpu_time()-t;
  ends[index]=wall_time();
  if(!bound&&sched_setaffinity(0,sizeof(previous),&previous))error=1;
 }
 double end=wall_time();
 metrics[0]=compute_begin-begin;metrics[1]=end-compute_begin;metrics[2]=cpu[0]+cpu[1]+cpu[2]+cpu[3];
 double compute_end=*std::max_element(ends,ends+4);
 metrics[3]=compute_begin-pack_begin;metrics[4]=compute_end-compute_begin;metrics[5]=end-compute_end;metrics[6]=pack_begin-begin;
 omp_set_dynamic(old_dynamic);
 return error?-1:0;
}
