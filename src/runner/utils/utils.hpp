#pragma once

#include <fstream>
#include <vector>
#include <iomanip>
#include <type_traits> // 用于类型检查
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept> // 用于异常处理

// 函数模板：支持任意元素类型的vector
template <typename T>
void savetxt(const std::string& filename, 
            const std::vector<T>& data,
            char delimiter = ' ', 
            int precision = 6)  // 默认精度调整为6位
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 保存流的原始格式状态
    std::ios_base::fmtflags original_flags = outfile.flags();

    // 仅对浮点类型启用科学计数法
    if constexpr (std::is_floating_point_v<T>) {
        outfile << std::scientific << std::setprecision(precision);
    }

    // 优化输出：先输出第一个元素，再循环输出后续元素
    if (!data.empty()) {
        outfile << data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            outfile << delimiter << data[i];
        }
    }

    // 恢复流的原始格式状态
    outfile.flags(original_flags);
    outfile.close();
}



// 函数模板：读取文本文件到 vector<T>
template <typename T>
int readtxt(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return -1;
    }

    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;

        std::istringstream iss(line);
        T value;
        // 解析行中的每个数值
        while (iss >> value) {
            data.push_back(value);
            // 跳过分隔符（兼容逗号、分号等）
            if (iss.peek() == ',' || iss.peek() == ';') iss.ignore();
        }
    }
    file.close();
    return 0;
}

std::vector<float> linspace(float start, float end, std::size_t num_steps) {
    if (num_steps == 0) {
        return {}; // 返回空向量
    }
    if (num_steps == 1) {
        return {start}; // 如果只需要一个点，返回起始值
    }

    std::vector<float> result(num_steps);
    float step_size = (end - start) / (num_steps - 1); // 计算步长

    for (std::size_t i = 0; i < num_steps; ++i) {
        result[i] = start + i * step_size;
    }

    // 确保最后一个值精确等于 end，避免浮点数精度带来的误差
    result.back() = end;

    return result;
}