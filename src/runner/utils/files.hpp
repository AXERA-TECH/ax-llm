#ifndef _FILES_H_
#define _FILES_H_

#include <sys/stat.h>  
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

bool is_directory(const std::string& path);
bool is_file(const std::string& path);
std::vector<std::string> list_files(const std::string& directory);


#endif