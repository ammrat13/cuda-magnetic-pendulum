#ifndef KERN_GPU_H
#define KERN_GPU_H

#include "kern_impl.cuh"


namespace kern::gpu {

    __global__ void compute_gpu(RawState st, Params p);
    __device__ float4 state_dt(float4 state);

    namespace dts {
        __device__ float4 inv_sq(float g, float off, float2 center, float4 state);
        __device__ float4 spring(float k, float2 center, float4 state);
        __device__ float4 frict(float m, float4 state);
    };
};


#endif
