#pragma once
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

struct GeneratorConfig {
    int      n_packets          = 1000;
    int      n_subcarriers      = 64;
    double   start_time         = 1700000000.0;
    double   packet_rate        = 100.0;
    double   jitter_ms          = 2.0;
    double   drop_rate          = 0.01;
    unsigned seed               = 42;
    std::string out_file        = "esp_raw.csv";
};

class CSIGenerator {

    GeneratorConfig                                  cfg_;
    std::mt19937                                     rng_;
    std::uniform_int_distribution<int>               uniform_iq_;
    std::uniform_real_distribution<double>           uniform_drop_;
    std::normal_distribution<double>                 normal_jitter_;

public:
    explicit CSIGenerator(const GeneratorConfig& cfg) 
        : cfg_(cfg)
        , rng_(cfg.seed)
        , uniform_iq_(-60, 60)
        , uniform_drop_(0.0, 1.0)
        , normal_jitter_(0.0, 1.0) 
    { }

    std::vector<int> generate_packet () {
        std::vector<int> re_im(cfg_.n_subcarriers * 2);
        for (int i = 0; i < cfg_.n_subcarriers * 2; ++i)
            re_im[i] = uniform_iq_(rng_);
        return re_im;
    }

    double next_timestamp (double prev_time) {
        if (uniform_drop_(rng_) < cfg_.drop_rate)
            return -1.0;
        double interval = 1.0 / cfg_.packet_rate + (cfg_.jitter_ms / 1000.0) * normal_jitter_(rng_) * 0.5;

        return prev_time + std::max(interval, 0.001);
    }

    void write(const std::string& path) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Не могу открыть: " + path);
        }

        double t = cfg_.start_time;
        int written = 0, dropped = 0;

        for (int i = 0; i < cfg_.n_packets; ++i) {
            double next_t = next_timestamp(t);
            if (next_t < 0.0) {
                t += 1.0 / cfg_.packet_rate;
                ++dropped;
                continue;
            }
            t = next_t;

            auto re_im = generate_packet();

            out << std::fixed;
            out.precision(3);
            out << t << ":[";
            for (int j = 0; j < re_im.size(); ++j) {
                if (j > 0) out << ",";
                out << re_im[j];
            }
            out << "]\n";

            ++written;
        }

        std::cout << "Записано: " << written << " | Потеряно: " << dropped << " | Файл: " << path << "\n";
    }
};