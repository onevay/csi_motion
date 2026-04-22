#pragma once
#include <vector>

struct Output_structure {   
    std::vector<std::vector<std::vector<double>>> packet;
    int label = 0;
    int person_id = 0;
};
