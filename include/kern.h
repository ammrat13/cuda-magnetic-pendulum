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

    struct StateElem {
        Vec2D pos;
        Vec2D vel;
    };

    class Kern {
        public:
            Kern(size_t res, Vec2D top, Vec2D bot);
            ~Kern();
            std::unique_ptr<const StateElem[]> getState() const;
        private:
            class KernImpl;
            const std::unique_ptr<KernImpl> kern_impl;
    };
};


#endif
