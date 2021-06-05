#ifndef KERN_H
#define KERN_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>


namespace kern {

    struct Vec2D {
        float x;
        float y;
    };


    struct Params {
        const size_t resolution;
        const float step_size;

        const Vec2D top_corner;
        const Vec2D bot_corner;

        size_t pixels() const { return this->resolution * this->resolution; };
    };


    class State {
        public:

            struct StateElem {
                Vec2D pos;
                Vec2D vel;
            };

            const size_t resolution;

            State(size_t res);
            StateElem& operator() (size_t r, size_t c);
            StateElem const& operator() (size_t r, size_t c) const;

        private:
            std::unique_ptr<StateElem[]> data;
            void check_bounds(size_t r, size_t c) const;
    };


    class Kern {
        public:

            Kern(Params p);
            ~Kern();

            void compute(size_t iters);
            State get_state() const;

        private:
            class KernImpl;
            const std::unique_ptr<KernImpl> kern_impl;
    };
};


#endif
