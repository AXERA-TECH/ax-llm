#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

class Num2Word_EN
{
private:
    const std::vector<std::string> ones = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"};

    const std::vector<std::string> tens = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

    const std::vector<std::string> scales = {
        "", "thousand", "million", "billion"};

    // 内部递归函数，处理正整数
    std::string process(long long n)
    {
        if (n < 20)
        {
            return ones[n];
        }

        if (n < 100)
        {
            std::string res = tens[n / 10];
            if (n % 10 != 0)
            {
                res += "-" + ones[n % 10];
            }
            return res;
        }

        if (n < 1000)
        {
            std::string res = ones[n / 100] + " hundred";
            if (n % 100 != 0)
            {
                res += " and " + process(n % 100);
            }
            return res;
        }

        // 处理千、百万、十亿 (int 范围最大到 20亿左右，所以处理到 billion 即可)
        for (int i = 3; i >= 1; --i)
        {
            long long scaleVal = std::pow(1000, i);
            if (n >= scaleVal)
            {
                std::string res = process(n / scaleVal) + " " + scales[i];
                long long remainder = n % scaleVal;

                if (remainder > 0)
                {
                    // 英文规则：如果余数小于100，通常用 "and" 连接 (如: one thousand and five)
                    // 如果余数较大，通常用逗号 (如: one thousand, two hundred)
                    if (remainder < 100)
                    {
                        res += " and " + process(remainder);
                    }
                    else
                    {
                        res += ", " + process(remainder);
                    }
                }
                return res;
            }
        }

        return ""; // Should not be reached
    }

public:
    std::string convert(int num)
    {
        if (num == 0)
            return "zero";

        // 使用 long long 防止 -INT_MIN 溢出
        long long val = num;
        std::string prefix = "";

        if (val < 0)
        {
            prefix = "minus ";
            val = -val;
        }

        return prefix + process(val);
    }

    std::string operator[](int num)
    {
        return convert(num);
    }
};