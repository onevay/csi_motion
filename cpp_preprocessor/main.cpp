#include "main.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: preprocessor dev1.data dev2.data dev3.data\n";
        return 1;
    }

    std::vector<std::string> errorLog;
    std::vector<std::vector<double>> result(3);

    for (int i = 0; i < 3; ++i) {
        std::vector<CSIPacket> packets = parseCSIFile_csv(argv[i + 1], errorLog);
        csi_calc_amplitude(packets);

        for (auto& pkt : packets) {
            median_filter(pkt.amplitudes);
        }

        for (const auto& pkt : packets) {
            result[i].insert(result[i].end(), pkt.amplitudes.begin(), pkt.amplitudes.end());
        }
    }
    
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "[";
    for (int dev = 0; dev < 3; ++dev) {
        std::cout << "[";
        for (int k = 0; k < static_cast<int>(result[dev].size()); ++k) {
            std::cout << result[dev][k];
            if (k + 1 < static_cast<int>(result[dev].size())) {
                std::cout << ",";
            }
        }
        std::cout << "]";
        if (dev < 2) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;
    return 0;
}