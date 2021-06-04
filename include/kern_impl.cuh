#ifndef KERN_IMPL_H
#define KERN_IMPL_H

#include "kern.h"


namespace kern {

    struct RawState {
        size_t pitch;
        float4 *data;
    };


    class Kern::KernImpl {

        public:

            KernImpl(size_t res, Vec2D top, Vec2D bot);
            ~KernImpl();

            std::unique_ptr<const StateElem[]> getState() const;

        private:

            const size_t resolution;
            const size_t pixels;
            const Vec2D top_corner;
            const Vec2D bot_corner;

            RawState hos_state;
            RawState dev_state;

            void compute();

    };
}


#endif
