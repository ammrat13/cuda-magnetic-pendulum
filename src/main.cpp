#include <fstream>

#include "kern.h"
#include "img.h"

const size_t RES   = 500;
const size_t ITERS = 100;
const float H      = 0.002;

const kern::Vec2D TOP_CORNER = {0.0, 1.0};
const kern::Vec2D BOT_CORNER = {1.0, 0.0};


int main() {

    kern::Params params = { RES, H, TOP_CORNER, BOT_CORNER };
    kern::Kern kernel{ params };

    kernel.compute(ITERS);
    auto result = kernel.getState();


    std::ofstream out;
    out.open("out.pbm", std::ios::out | std::ios::trunc);

    img::write::pbm(out, params, result.get(), img::color_funcs::sign_y);
}
