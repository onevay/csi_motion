#pragma once
#include <vector>

struct CSIPacket {
    double timestamp;
    std::vector<int> raw_re_im;
    std::vector<double> amplitudes;
};

struct CSIFile {
    int person_id = 0;
    int label = 0;
    int test = 0;
    int number_dev = 0;
    std::vector<CSIPacket> packets;
};