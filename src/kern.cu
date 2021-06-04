#include "kern.h"
#include "kern_impl.cuh"


kern::Kern::Kern(kern::Params p) : kern_impl(new kern::Kern::KernImpl{p}) {}
kern::Kern::~Kern() = default;

std::unique_ptr<const kern::State> kern::Kern::getState() const {
    return kern_impl->getState();
}
