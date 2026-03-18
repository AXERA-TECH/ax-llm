#include "files.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

bool is_directory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

bool is_file(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

std::vector<std::string> list_files(const std::string& directory) {
    std::vector<std::string> files;
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        std::cerr << "failed to open directory: " << directory << "\n";
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (ec) {
            std::cerr << "failed to iterate directory: " << directory << "\n";
            return files;
        }
        if (entry.is_regular_file(ec) && !ec) {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}
