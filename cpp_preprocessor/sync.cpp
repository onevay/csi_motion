#include "main.hpp"

std::vector<Output_structure> sync (const std::vector<CSIFile>& files) {
    std::map<std::tuple<int, int, int>, std::map<int, const CSIFile*>> grouped;

    for (const auto& file : files) {
        auto key = std::make_tuple(file.person_id, file.test, file.label);
        grouped[key][file.number_dev] = &file;
    }

    std::map<int, std::map<int, Output_structure>> by_person;
    for (int pid = 1; pid <= 4; ++pid)
        for (int lb = 0; lb <= 3; ++lb) {
            by_person[pid][lb].person_id = pid;
            by_person[pid][lb].label = lb;
        }

    for (const auto& [key, devMap] : grouped) {
        auto& [person_id, test, label] = key;

        if (devMap.find(1) == devMap.end() || devMap.find(2) == devMap.end() || devMap.find(3) == devMap.end()) {
            continue;
        }

        std::vector<std::vector<double>> packet_triplet(3);
        
        for (int dev = 1; dev <= 3; ++dev) {
            const CSIFile* f = devMap.at(dev);
            std::vector<double>& devAmps = packet_triplet[dev - 1];

            devAmps.reserve(5200);

            int packets_to_copy = std::min(100, static_cast<int>(f->packets.size()));
            
            for (int i = 0; i < packets_to_copy; ++i) {
                devAmps.insert(devAmps.end(), f->packets[i].amplitudes.begin(), f->packets[i].amplitudes.end());
            }

            if (devAmps.size() < 5200) {
                devAmps.resize(5200, 0.0);
            } else if (devAmps.size() > 5200) {
                devAmps.resize(5200);
            }
        }
        by_person[person_id][label].packet.push_back(std::move(packet_triplet));
    }

    std::vector<Output_structure> result;
    for (auto& p_map : by_person) {
        for (auto& l_map : p_map.second) {
            if (!l_map.second.packet.empty()) {
                result.push_back(std::move(l_map.second));
            }
        }
    }

    return result;
}