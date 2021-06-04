#include "kern_impl.cuh"
#include "kern_gpu.cuh"


kern::Kern::KernImpl::KernImpl(size_t res, Vec2D top, Vec2D bot):
    resolution{res},
    pixels{res*res},
    top_corner{top},
    bot_corner{bot}
{

    this->hos_state.pitch = this->resolution * sizeof(float4);
    this->hos_state.data = new float4[this->pixels];

    cudaError_t alloc_res = cudaMallocPitch(
        &this->dev_state.data,
        &this->dev_state.pitch,
        this->resolution * sizeof(float4),
        this->resolution
    );
    if(alloc_res != cudaSuccess) {
        throw;
    }

    this->compute();
}

kern::Kern::KernImpl::~KernImpl() {
    delete[] this->hos_state.data;
    cudaFree(this->dev_state.data);
}


std::unique_ptr<const kern::StateElem[]> kern::Kern::KernImpl::getState() const {

    std::unique_ptr<kern::StateElem[]> ret(new kern::StateElem[this->pixels]);

    for(size_t r = 0; r < this->resolution; r++) {
        for(size_t c = 0; c < this->resolution; c++) {
            float4 cur = this->hos_state.data[r * this->hos_state.pitch/sizeof(float4) + c];
            ret[r * this->resolution + c] = {
                {cur.x, cur.y},
                {cur.z, cur.w}
            };
        }
    }

    return ret;
}


void kern::Kern::KernImpl::compute() {
    const size_t parallelism = 32;

    const dim3 num_blocks(this->resolution/parallelism + 1, this->resolution, 1);
    const dim3 th_per_blk(parallelism, 1, 1);

    kern::gpu::compute_gpu<<<num_blocks, th_per_blk>>>(
        this->dev_state,
        this->resolution,
        this->top_corner, this->bot_corner
    );

    cudaError_t memcpy_res = cudaMemcpy2D(
        this->hos_state.data,
        this->hos_state.pitch,
        this->dev_state.data,
        this->dev_state.pitch,
        this->resolution * sizeof(float4),
        this->resolution,
        cudaMemcpyDeviceToHost
    );
    if(memcpy_res != cudaSuccess) {
        throw;
    }
}
