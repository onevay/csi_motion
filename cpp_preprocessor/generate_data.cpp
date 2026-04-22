#include "generate_data.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Введите название хотя бы 1 файла для генерации!\n";
        return 0;
    }

    for (int i = 1 ; i < argc ; ++i) {
        GeneratorConfig cfg;
        cfg.out_file = argv[i];
        CSIGenerator gen(cfg);
        gen.write(cfg.out_file);
    }

}