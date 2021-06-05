#include "img.h"


const img::Color img::Color::WHITE = {0xff, 0xff, 0xff};
const img::Color img::Color::BLACK = {0x00, 0x00, 0x00};


img::Color img::color_funcs::sign_y(const kern::State::StateElem& s) {
    return s.pos.y >= 0
        ? img::Color::BLACK
        : img::Color::WHITE;
}
