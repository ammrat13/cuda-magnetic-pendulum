#ifndef IMG_H
#define IMG_H

#include <cstdint>
#include <fstream>

#include "kern.h"


namespace img {

    struct Color {
        uint8_t r;
        uint8_t g;
        uint8_t b;

        static const Color WHITE;
        static const Color BLACK;
    };


    using ColorFunc = Color (const kern::StateElem&);
    namespace color_funcs {
        ColorFunc sign_y;
    };

    using WriteFunc = void (
        std::ostream&,
        const kern::Params&,
        const kern::State,
        const ColorFunc&);
    namespace write {
        WriteFunc pbm;
    }
};


#endif
