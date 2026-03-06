#pragma once

#include <string>
#include <vector>

bool is_directory(const std::string& path);
bool is_file(const std::string& path);
std::vector<std::string> list_files(const std::string& directory);

