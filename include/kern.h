#ifndef KERN_H
#define KERN_H

#include <cstddef>
#include <cstdint>

#include <memory>


namespace kern {

    struct Vec2D {
        float x;
        float y;
    };


    struct Params {
        const size_t resolution;
        const size_t iterations;
        const float step_size;

        const Vec2D top_corner;
        const Vec2D bot_corner;

        size_t pixels() const { return this->resolution * this->resolution; };
    };

    struct StateElem {
        Vec2D pos;
        Vec2D vel;
    };
    using State = StateElem[];


    class Kern {
        public:
            Kern(Params p);
            ~Kern();
            std::unique_ptr<const State> getState() const;
        private:
            class KernImpl;
            const std::unique_ptr<KernImpl> kern_impl;
    };
};


#endif
