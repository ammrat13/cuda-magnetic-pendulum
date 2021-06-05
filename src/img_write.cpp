#include "img.h"


void img::write::pbm(
    std::ostream& file,
    const kern::Params& p,
    const kern::State& s,
    const img::ColorFunc& cf
) {

    file << "P3\n";
    file << p.resolution << " " << p.resolution << "\n";
    file << "255\n";

    for(size_t r = 0; r < p.resolution; r++) {
        for(size_t c = 0; c < p.resolution; c++) {
            img::Color res = cf(s(r, c));
            file << res.r << " " << res.g << " " << res.b << "\n";
        }
    }

    file.flush();
}
