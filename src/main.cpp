#include <iostream>
#include <fstream>

#include "kern.h"


const size_t RES   = 500;
const size_t ITERS = 10000;
const float H      = 0.002;

bool color_func(kern::StateElem e) {
    return e.pos.y < 0;
}

int main() {

    std::ofstream out;
    out.open("out.pbm", std::ios::out | std::ios::trunc);

    out << "P1" << std::endl;
    out << RES << " " << RES << std::endl;

    kern::Kern kernel{ { RES, ITERS, H, {0.0, 1.0}, {1.0, 0.0} } };
    std::unique_ptr<const kern::StateElem[]> ret = kernel.getState();

    for(size_t r = 0; r < RES; r++) {
        for(size_t c = 0; c < RES; c++) {
            out << color_func(ret[r*RES + c]) << " ";
        }
        out << std::endl;
    }
}
