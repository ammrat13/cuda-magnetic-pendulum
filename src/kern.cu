#include "kern.h"
#include "kern_impl.cuh"


kern::Kern::Kern(kern::Params p) : kern_impl(new kern::Kern::KernImpl{p}) {}
kern::Kern::~Kern() = default;

void kern::Kern::compute(size_t iters) {
    this->kern_impl->compute(iters);
}
kern::State kern::Kern::get_state() const {
    return this->kern_impl->get_state();
}
