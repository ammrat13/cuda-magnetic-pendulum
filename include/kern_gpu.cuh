#ifndef KERN_GPU_H
#define KERN_GPU_H

#include "kern_impl.cuh"


namespace kern::gpu {

    __global__ void compute_gpu(
        RawState st,
        Params p,
        unsigned int iters,
        bool first_call);
    __device__ float4 state_dt(float4 state);

    namespace dts {
        __forceinline__ __device__ float2 inv_sq(float g, float off, float2 center, float4 state);
        __forceinline__ __device__ float2 spring(float k, float2 center, float4 state);
        __forceinline__ __device__ float2 frict(float m, float4 state);
    };
};


#endif
