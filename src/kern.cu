#include "kern.h"
#include "kern_impl.cuh"


kern::Kern::Kern(size_t res, Vec2D top, Vec2D bot) :
    kern_impl(new kern::Kern::KernImpl{res, top, bot})
{}

kern::Kern::~Kern() = default;


std::unique_ptr<const kern::StateElem[]> kern::Kern::getState() const {
    return kern_impl->getState();
}
