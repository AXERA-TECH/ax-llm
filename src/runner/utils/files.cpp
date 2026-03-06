#include "files.hpp"

#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

bool is_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

bool is_file(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISREG(st.st_mode);
}

std::vector<std::string> list_files(const std::string& directory) {
    std::vector<std::string> files;
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        std::cerr << "failed to open directory: " << directory << "\n";
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string full_path = directory + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.push_back(full_path);
        }
    }
    closedir(dir);

    std::sort(files.begin(), files.end());
    return files;
}

