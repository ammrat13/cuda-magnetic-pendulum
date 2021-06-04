#include "kern_gpu.cuh"


__device__ inline float4 operator*(const float& a, const float4& b) {
    return make_float4(
        a*b.x,
        a*b.y,
        a*b.z,
        a*b.w
    );
}

__device__ inline float4 operator+(const float4& a, const float4& b) {
    return make_float4(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    );
}


__global__ void kern::gpu::compute_gpu(
    kern::RawState st,
    kern::Params p
) {

    unsigned int rowIdx = blockIdx.y;
    unsigned int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(colIdx >= p.resolution) {
        return;
    }

    float rowA = ((float) rowIdx + 0.5) / (float) p.resolution;
    float colA = ((float) colIdx + 0.5) / (float) p.resolution;

    float4 state = make_float4(
        (1-colA)*p.top_corner.x + colA*p.bot_corner.x,
        (1-rowA)*p.top_corner.y + rowA*p.bot_corner.y,
        0.0,
        0.0
    );

    const float h = p.step_size;
    for(unsigned int i = 0; i < p.iterations; i++) {
        float4 k1 = kern::gpu::state_dt(state);
        float4 k2 = kern::gpu::state_dt(state + h/2 * k1);
        float4 k3 = kern::gpu::state_dt(state + h/2 * k2);
        float4 k4 = kern::gpu::state_dt(state + h * k3);
        state = state + h/6 * (k1 + 2*k2 + 2*k3 + k4);
    }

    st.data[rowIdx * st.pitch/sizeof(float4) + colIdx] = state;
}

__device__ float4 kern::gpu::state_dt(float4 state) {
    float4 top_inv_sq = kern::gpu::dts::inv_sq(1.5, 0.15, make_float2(0.0, 0.5), state);
    float4 bot_inv_sq = kern::gpu::dts::inv_sq(1.5, 0.15, make_float2(0.0,-0.5), state);
    float4 mid_spring = kern::gpu::dts::spring(0.5, make_float2(0.0, 0.0), state);

    float4 frict_force = kern::gpu::dts::frict(0.1, state);

    return make_float4(state.z, state.w, 0.0, 0.0)
        + top_inv_sq
        + bot_inv_sq
        + mid_spring
        + frict_force;
}


__device__ float4 kern::gpu::dts::inv_sq(float g, float off, float2 center, float4 state) {
    float2 d = make_float2(state.x-center.x, state.y-center.y);
    float mag = pow(d.x*d.x + d.y*d.y + off*off, -1.5);
    return make_float4(0.0, 0.0, -g*mag*d.x, -g*mag*d.y);
}

__device__ float4 kern::gpu::dts::spring(float k, float2 center, float4 state) {
    float2 d = make_float2(state.x-center.x, state.y-center.y);
    return make_float4(0.0, 0.0, -k*d.x, -k*d.y);
}

__device__ float4 kern::gpu::dts::frict(float m, float4 state) {
    return make_float4(0.0, 0.0, -m*state.z, -m*state.w);
}
