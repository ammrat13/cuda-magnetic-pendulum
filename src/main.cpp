#include <fstream>

#include "kern.h"
#include "img.h"


const size_t RES   = 500;
const size_t ITERS = 100000;
const float H      = 0.002;

const kern::Vec2D TOP_CORNER = {0.0, 1.0};
const kern::Vec2D BOT_CORNER = {1.0, 0.0};

const std::string OUT_FILE_NAME = "out.pbm";


int main() {

    kern::Params params = { RES, H, TOP_CORNER, BOT_CORNER };
    kern::Kern kernel{ params };

    kernel.compute(ITERS);
    kern::State result = kernel.get_state();


    std::ofstream out;
    out.open(OUT_FILE_NAME, std::ios::out | std::ios::trunc);

    img::write::pbm(out, params, result, img::color_funcs::sign_y);
}
