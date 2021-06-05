#include "kern.h"


kern::State::State(size_t res): resolution{res} {
    this->data = std::make_unique<kern::State::StateElem[]>(
        this->resolution * this->resolution
    );
}


void kern::State::check_bounds(size_t r, size_t c) const {
    if(r >= this->resolution || c >= this->resolution) {
        throw std::out_of_range("Out of Range error: kern::State");
    }
}

kern::State::StateElem& kern::State::operator() (size_t r, size_t c) {
    this->check_bounds(r, c);
    return this->data[r * this->resolution + c];
}

kern::State::StateElem const& kern::State::operator() (size_t r, size_t c) const {
    this->check_bounds(r, c);
    return this->data[r * this->resolution + c];
}
