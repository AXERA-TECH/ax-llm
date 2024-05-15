#include "memory_utils.hpp"

bool file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

bool read_file(const std::string &path, std::vector<char> &data)
{

    std::fstream fs(path, std::ios::in | std::ios::binary);

    if (!fs.is_open())
    {
        return false;
    }

    // get file size
    fs.seekg(0, std::ios::end);
    size_t file_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    if (file_size == 0)
    {
        return false;
    }

    data.resize(file_size);
    fs.read(data.data(), file_size);
    // data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();

    return true;
}

bool read_file(const std::string &path, char **data, size_t *len)
{
    FILE *fp = fopen(path.c_str(), "rb");

    if (!fp)
    {
        return false;
    }

    fseek(fp, 0, SEEK_END);

    *len = ftell(fp);

    fseek(fp, 0, SEEK_SET);

    *data = new char[*len];

    fread(*data, *len, 1, fp);

    fclose(fp);

    return true;
}
