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

            void compute(size_t iters);
            State get_state() const;

        private:

            const Params p;
            bool first_call;

            RawState hos_state;
            RawState dev_state;

    };
}


#endif
