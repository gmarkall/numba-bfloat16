#include "cuda_bf16.h"

__device__ __nv_bfloat16 f(int x) {
  return __int2bfloat16_rd(x);
}

__device__ __nv_bfloat16 f(float x) {
  return __float2bfloat16(x);
}


__device__ __nv_bfloat16 add(__nv_bfloat16 x, __nv_bfloat16 y) {
  return __hadd(x, y);
}

__device__ __nv_bfloat16 mul(__nv_bfloat16 x, __nv_bfloat16 y) {
  return __hmul(x, y);
}


__device__ __nv_bfloat16 muladd(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 z) {
  return __hfma(x, y, z);
}
