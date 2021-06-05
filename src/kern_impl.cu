#include "kern_impl.cuh"
#include "kern_gpu.cuh"


kern::Kern::KernImpl::KernImpl(kern::Params p): p{p}, first_call{true} {

    this->hos_state.pitch = this->p.resolution * sizeof(float4);
    this->hos_state.data = new float4[this->p.pixels()];

    auto alloc_res = cudaMallocPitch(
        &this->dev_state.data,
        &this->dev_state.pitch,
        this->p.resolution * sizeof(float4),
        this->p.resolution
    );
    if(alloc_res != cudaSuccess) {
        throw;
    }
}

kern::Kern::KernImpl::~KernImpl() {
    delete[] this->hos_state.data;
    cudaFree(this->dev_state.data);
}


std::unique_ptr<const kern::State> kern::Kern::KernImpl::getState() const {

    auto ret = std::make_unique<kern::State>(this->p.pixels());

    for(size_t r = 0; r < this->p.resolution; r++) {
        for(size_t c = 0; c < this->p.resolution; c++) {
            float4 cur = this->hos_state.data[
                r * this->hos_state.pitch/sizeof(float4) + c
            ];
            ret[r * this->p.resolution + c] = {
                {cur.x, cur.y},
                {cur.z, cur.w}
            };
        }
    }

    return ret;
}


void kern::Kern::KernImpl::compute(size_t iters) {
    const size_t parallelism = 256;

    const dim3 num_blocks(
        (this->p.resolution + parallelism - 1) / parallelism,
        this->p.resolution,
        1
    );
    const dim3 th_per_blk(parallelism, 1, 1);

    kern::gpu::compute_gpu<<<num_blocks, th_per_blk>>>(
        this->dev_state,
        this->p,
        iters,
        this->first_call
    );

    auto memcpy_res = cudaMemcpy2D(
        this->hos_state.data,
        this->hos_state.pitch,
        this->dev_state.data,
        this->dev_state.pitch,
        this->p.resolution * sizeof(float4),
        this->p.resolution,
        cudaMemcpyDeviceToHost
    );
    if(memcpy_res != cudaSuccess) {
        throw;
    }

    this->first_call = false;
}
