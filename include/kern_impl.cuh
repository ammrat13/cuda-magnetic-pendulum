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

            KernImpl(Params p);
            ~KernImpl();

            std::unique_ptr<const State> getState() const;

        private:

            const Params p;

            RawState hos_state;
            RawState dev_state;

            void compute();

    };
}


#endif
