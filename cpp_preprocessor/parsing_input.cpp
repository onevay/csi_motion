#include "main.hpp"

std::vector<CSIPacket> parseCSIFile_csv (const std::string& path, std::vector<std::string>& errorLog) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }

    std::vector<CSIPacket> packets;
    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { 
            first_line = false; 
            continue; 
        }

        if (line.empty()) {
            continue; 
        }

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::size_t q1 = line.find('"');
        std::size_t q2 = line.rfind('"');

        if (q1 == std::string::npos || q2 == q1) {
            errorLog.push_back(path + " == bad line (no quoted array)");
            continue;
        }

        std::string arrStr = line.substr(q1 + 1, q2 - q1 - 1);
        std::size_t start = arrStr.find('[');
        std::size_t end   = arrStr.rfind(']');

        if (start == std::string::npos || end == std::string::npos) {
            errorLog.push_back(path + " == bad array");
            continue;
        }

        std::string inner = arrStr.substr(start + 1, end - start - 1);
        std::istringstream ss(inner);
        std::string token;
        std::vector<int> reIm;
        try {
            while (std::getline(ss, token, ','))
                if (!token.empty())
                    reIm.push_back(std::stoi(token));
        } catch (...) {
            errorLog.push_back(path + " == parse error");
            continue;
        }

        double ts = 0.0;
        try {
            std::size_t comma = line.find(',');
            if (comma != std::string::npos)
                ts = parseTimestamp(line.substr(0, comma));
        } catch (...) {}

        CSIPacket pkt;
        pkt.timestamp = ts;
        pkt.raw_re_im = std::move(reIm);
        packets.push_back(std::move(pkt));
    }
    return packets;
}

double parseTimestamp (const std::string& dtStr) {
    int day, month, year, hour, min, sec, ms;
    int hour_1, min_1;
    char sign;

    int n = std::sscanf(dtStr.c_str(),
        "%d.%d.%d %d:%d:%d.%d %c%d:%d",
        &day, &month, &year, &hour, &min, &sec, &ms, &sign, &hour_1, &min_1);

    if (n != 10) 
        throw std::runtime_error("Failed to parse timestamp: " + dtStr);

    auto isLeap = [](int y) { 
        return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0); 
    };

    static const int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    long long days = 0;

    for (int y = 1970; y < year; ++y)
        days += isLeap(y) ? 366 : 365;

    for (int m = 1; m < month; ++m) {
        days += daysInMonth[m - 1];
        if (m == 2 && isLeap(year)) {
            ++days;
        }
    }

    days += day - 1;

    long long total_sec = days * 86400LL + hour * 3600LL + min * 60LL + sec;

    if (n != 10) 
        throw std::runtime_error("Failed to parse timestamp: " + dtStr);

    int offsetSec = hour_1 * 3600 + min_1 * 60;

    if (sign == '+') {
        total_sec -= offsetSec;
    } else {
        total_sec += offsetSec;
    }

    return static_cast<double>(total_sec) + ms / 1000.0;
}

static std::optional<std::vector<int>> extractArray (const std::string& line) {
    std::size_t start = line.find('\"');

    if (start == std::string::npos) {
        return std::nullopt;
    }

    std::size_t end = line.find('\"', start + 1);

    if (end == std::string::npos) {
        return std::nullopt;
    }

    std::string content = line.substr(start + 1, end - start - 1);

    if (content.front() == '[' && content.back() == ']') {
        content = content.substr(1, content.size() - 2);
    }

    std::vector<int> res;
    std::stringstream ss(content);
    std::string val;
    while (std::getline(ss, val, ',')) {
        try {
            res.push_back(std::stoi(val));
        } catch (...) {}
    }
    return res;
}

std::vector<CSIPacket> parseCSIFile (const std::string& path, std::vector<std::string>& errorLog) {
    std::vector<CSIPacket> fileData;
    std::ifstream f(path);
    if (!f.is_open()) {
        errorLog.push_back("Could not open file: " + path);
        return fileData;
    }

    std::string line;
    std::vector<std::string> lines;

    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    auto logSkip = [&](int ln, const std::string& reason) {
        errorLog.push_back("File: " + path + " | Line: " + std::to_string(ln) + " | Reason: " + reason);
    };

    for (int i = 0; i < lines.size(); ++i) {
        const std::string& line = lines[i];
        int lineNum = i + 1;

        std::size_t csiPos = line.find("CSI_DATA");

        if (csiPos == std::string::npos) {
            logSkip(lineNum, "no CSI_DATA tag");
            continue;
        }

        std::string dStr = line.substr(0, csiPos);
        double ts;

        try {
            ts = parseTimestamp(dStr);
        } catch (...) {
            logSkip(lineNum, "bad timestamp: " + dStr);
            continue;
        }

        std::optional<std::vector<int>> reImOpt = extractArray(line);

        if (!reImOpt.has_value()) {
            logSkip(lineNum, "no quoted array — skip line to maintain sync");
            continue; 
        }

        CSIPacket pkt;
        pkt.timestamp = ts;
        pkt.raw_re_im = std::move(reImOpt.value());
        fileData.push_back(std::move(pkt));
    }

    return fileData;
}