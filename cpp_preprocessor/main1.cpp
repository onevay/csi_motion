#include "main.hpp"
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    fs::path root = "/Users/rech/Downloads/wifi_data_set";
    std::vector<CSIFile> result_files;
    std::vector<std::string> errorLog;

    int counter_dev1 = 0;
    int counter_dev2 = 0;
    int counter_dev3 = 0;

    for (auto& person : fs::directory_iterator(root)) {
        if (!person.is_directory()) {
            continue;
        }

        std::string person_name = person.path().filename().string();
        int person_id = 0;
        std::size_t person_underscore = person_name.rfind('_');
        person_id = std::stoi(person_name.substr(person_underscore+1));

        for (auto& label_dir : fs::directory_iterator(person)) {
            if (!label_dir.is_directory()) {
                continue;
            }

            std::string label_name = label_dir.path().filename().string();
            int label_val = 0;
            std::size_t label_underscore = label_name.rfind('_');
            label_val = std::stoi(label_name.substr(label_underscore + 1));

            for (auto& test : fs::directory_iterator(label_dir)) {
                if (!test.is_directory()) {
                    continue; 
                }

                std::string test_name = test.path().filename().string();
                int test_val = 0;
                std::size_t test_underscore = test_name.rfind('_');
                test_val = std::stoi(test_name.substr(test_underscore + 1));

                for (auto& file : fs::directory_iterator(test)) {
                    if (!file.is_regular_file()) continue;

                    std::string filename = file.path().filename().string();
                    if (filename.find("test") != 0 ||
                        filename.find("__dev") == std::string::npos) {
                        continue;
                    }

                    int dev_val = 0;
                    std::size_t devPos = filename.find("__dev");
                    if (devPos != std::string::npos) {
                        std::size_t numStart = devPos + 5;
                        std::size_t numEnd = numStart;

                        while (numEnd < filename.size() && std::isdigit(filename[numEnd])) {
                            ++numEnd;
                        }

                        dev_val = std::stoi(filename.substr(numStart, numEnd - numStart));
                    }

                    std::string filepath = file.path().string();

                    CSIFile csi_file;

                    switch (dev_val) {
                        case 1:
                            ++counter_dev1;
                            break;
                        case 2:
                            ++counter_dev2;
                            break;
                        case 3:
                            ++counter_dev3;
                            break;
                    }

                    csi_file.person_id  = person_id;
                    csi_file.label      = label_val;
                    csi_file.test       = test_val;
                    csi_file.number_dev = dev_val;
                    csi_file.packets    = parseCSIFile(filepath, errorLog);
                    csi_calc_amplitude(csi_file.packets);

                    std::cout << "[OK] " << filepath << " | dev: "     << dev_val
                    << " | label: "   << label_val << " | packets: " << csi_file.packets.size() << "\n";

                    result_files.push_back(std::move(csi_file));
                }
            }
        }
    }

    if (!errorLog.empty()) {
        std::cerr << "\n=== PARSE ERRORS ===\n";
        for (const auto& entry : errorLog)
            std::cerr << entry << "\n";
        std::cerr << "=== TOTAL: " << errorLog.size() << " error(s) ===\n";
    }

    for (const auto& f : result_files) {
        for (const auto& v : f.packets) {
            if (v.amplitudes.size() != 52)
                std::cout << "error amplitude: " << v.amplitudes.size() << "\t";
            if (v.raw_re_im.size() != 128)
                std::cout << "error reIm: " << v.raw_re_im.size() << "\t";
        }
    }

    std::cout << "dev1: " << counter_dev1 << "\tdev2: " << counter_dev2 << "\tdev3: " << counter_dev3 << "\n";

    for (auto& v : result_files) {
        for (auto& s : v.packets) {
            median_filter(s.amplitudes);
        }
    }

    std::vector<Output_structure> out_files;

    std::map<int, std::string> csv_paths = {
        {1, "/Users/rech/.vscode/yandex_studcamp/wifi_data_set_after_preprocessing_person_id_1.csv"},
        {2, "/Users/rech/.vscode/yandex_studcamp/wifi_data_set_after_preprocessing_person_id_2.csv"},
        {3, "/Users/rech/.vscode/yandex_studcamp/wifi_data_set_after_preprocessing_person_id_3.csv"},
        {4, "/Users/rech/.vscode/yandex_studcamp/wifi_data_set_after_preprocessing_person_id_4.csv"},
    };

    out_files = sync(result_files);

    for (auto& out_struct : out_files) {
        for (auto& test_entry : out_struct.packet) {
            for (auto& dev_amplitudes : test_entry) {
                if (!dev_amplitudes.empty()) {
                    median_filter(dev_amplitudes);
                }
            }
        }
    }

    save_to_csv(out_files, csv_paths);
    return 0;
}