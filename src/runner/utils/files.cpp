#include <sys/stat.h>  
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

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
        std::cerr << "无法打开目录: " << directory << std::endl;
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        // 拼接完整路径并检查是否为普通文件
        std::string full_path = directory + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.push_back(full_path);
        }
    }
    closedir(dir);

    // 按文件名升序排序
    std::sort(files.begin(), files.end());
    return files;
}