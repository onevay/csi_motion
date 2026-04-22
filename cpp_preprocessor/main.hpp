#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <map>
#include <tuple>
#include <optional>
#include <iomanip>
#include <cctype>
#include <cmath>
#include <cstdio>
#include "csi_structs.hpp"
#include "output_struct.hpp"

void csi_calc_amplitude (std::vector<CSIPacket> &vec);

void median_filter (std::vector<double>& amplitudes);

void save_to_csv (const std::vector<Output_structure>& out_files, const std::map<int, std::string>& paths);

double parseTimestamp(const std::string& dtStr);

std::vector<CSIPacket> parseCSIFile (const std::string& path, std::vector<std::string>& errorLog);

std::vector<CSIPacket> parseCSIFile_csv(const std::string& path, std::vector<std::string>& errorLog);

std::vector<Output_structure> sync (const std::vector<CSIFile>& files);