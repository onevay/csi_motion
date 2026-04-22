#include "csi_structs.hpp"
#include "main.hpp"

void csi_calc_amplitude (std::vector<CSIPacket> &vec) {
    std::vector<int> system_amplitude = {0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37};

    for (auto &v : vec) {
        for (int i = 0 ; i < v.raw_re_im.size() / 2 ; ++i) {
            if (std::find(system_amplitude.begin(), system_amplitude.end(), i) != system_amplitude.end()) {
                continue;
            } else {
                std::pair<int, int> re_im = {v.raw_re_im[i * 2], v.raw_re_im[i * 2 + 1]};
                v.amplitudes.push_back(sqrt(re_im.first * re_im.first + re_im.second * re_im.second));
            }
        }
    }
}