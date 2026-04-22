#include "main.hpp"

void median_filter (std::vector<double>& amplitudes) {
    int size = amplitudes.size();

    if (size < 5) {
        return;
    }

    std::vector<double> window(5);
    
    for (int i = 0; i < 5; ++i) {
        window[i] = amplitudes[i];
    }

    std::nth_element(window.begin(), window.begin() + 2, window.end());
    double start_median = window[2];

    for (int i = 0; i < 5; ++i) {
        window[i] = amplitudes[size - 1 - i];
    }

    std::nth_element(window.begin(), window.begin() + 2, window.end());
    double end_median = window[2];

    std::vector<double> padded;
    padded.reserve(size + 4);

    padded.push_back(start_median);
    padded.push_back(start_median);

    for (double val : amplitudes) {
        padded.push_back(val);
    }

    padded.push_back(end_median);
    padded.push_back(end_median);

    std::vector<double> filtered;
    filtered.reserve(size);

    for (int i = 0; i < size; ++i) {
        window[0] = padded[i];
        window[1] = padded[i + 1];
        window[2] = padded[i + 2];
        window[3] = padded[i + 3];
        window[4] = padded[i + 4];
        
        std::nth_element(window.begin(), window.begin() + 2, window.end());
        filtered.push_back(window[2]);
    }

    amplitudes = std::move(filtered);
}