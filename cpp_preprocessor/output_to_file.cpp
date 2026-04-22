#include "main.hpp"

void save_to_csv (const std::vector<Output_structure>& out_files, const std::map<int, std::string>& paths) {

    std::map<int, std::vector<const Output_structure*>> by_person;
    for (const auto& v : out_files)
        by_person[v.person_id].push_back(&v);

    for (const auto& [person_id, structures] : by_person) {
        const std::string& filepath = paths.at(person_id);
        std::ofstream file(filepath);

        int counter = 1;

        for (const auto* v : structures) {
            for (const auto& test : v->packet) {
                file << counter << ":";

                file << "[";
                for (int dev = 0; dev < static_cast<int>(test.size()); ++dev) {
                    file << "[";
                    for (int k = 0; k < static_cast<int>(test[dev].size()); ++k) {
                        file << test[dev][k];
                        if (k + 1 < static_cast<int>(test[dev].size()))
                            file << ",";
                    }
                    file << "]";
                    if (dev + 1 < static_cast<int>(test.size()))
                        file << ",";
                }
                file << "]";

                file << ":" << v->label << "\n";
                ++counter;
            }
        }

        std::cout << "[SAVED] " << filepath << " | строк: " << (counter - 1) << "\n";
    }
}