#pragma once
#include <string>

class UTF8Filter
{
private:
    std::string utf8_buffer = "";

    // 辅助函数：计算字符串中“有效完整 UTF-8”部分的长度
    // 返回值是可以安全发送的字节数
    size_t get_valid_utf8_len(const std::string &str)
    {
        size_t len = str.length();
        if (len == 0)
            return 0;

        // 从字符串末尾开始回溯，检查最后一个字符是否完整
        // UTF-8 最大长度为 4 字节，所以最多回溯 4 步
        for (int i = 0; i < 4; ++i)
        {
            if (len <= i)
                break; // 已经回溯到头了

            unsigned char byte = static_cast<unsigned char>(str[len - 1 - i]);

            // 1. 如果是 ASCII (0xxxxxxx)，那就是完整的，结束在它后面
            if ((byte & 0x80) == 0)
            {
                // 如果回溯了 (i > 0)，说明后面跟着的字节是不合法的孤立后缀，
                // 但在流式场景下，我们通常假设数据流是合法的，只是还没发完。
                // 这里为了简单：如果最后一位是 ASCII，那它就是完整的边界。
                if (i == 0)
                    return len;
                // 如果 i > 0，说明后面有 i 个 continuation byte 找不到头，这属于流还没到齐
                // 继续找头
            }

            // 2. 检查是否是 Header Byte (11xxxxxx)
            if ((byte & 0xC0) == 0xC0)
            {
                int needed_extra = 0;
                if ((byte & 0xE0) == 0xC0)
                    needed_extra = 1; // 2-byte char
                else if ((byte & 0xF0) == 0xE0)
                    needed_extra = 2; // 3-byte char
                else if ((byte & 0xF8) == 0xF0)
                    needed_extra = 3; // 4-byte char

                // i 是当前 Header 后面实际跟的字节数
                if (i >= needed_extra)
                {
                    return len; // 完整了
                }
                else
                {
                    // 不完整，这个 Header 以及后面的 i 个字节都要留给下一次
                    return len - 1 - i;
                }
            }

            // 3. 如果是 Continuation Byte (10xxxxxx)，继续回溯找 Header
        }

        // 如果回溯 4 步都没找到 Header 或 ASCII，说明数据可能有问题
        // 为了防止死锁，我们假设全部发送（让 JSON 库去处理或报错，或者丢弃）
        // 但在流式拼接中，通常返回 0 等待更多数据
        return 0;
    }

public:
    std::string filter(const std::string &str)
    {
        utf8_buffer += str;
        size_t send_len = get_valid_utf8_len(utf8_buffer);
        if (send_len > 0)
        {
            std::string part_to_send = utf8_buffer.substr(0, send_len);

            if (send_len < utf8_buffer.size())
            {
                utf8_buffer = utf8_buffer.substr(send_len);
            }
            else
            {
                utf8_buffer.clear();
            }
            return part_to_send;
        }
        else
        {
            return std::string();
        }
    }
};
