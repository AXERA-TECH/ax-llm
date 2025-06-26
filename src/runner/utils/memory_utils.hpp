#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <errno.h>
#include <mntent.h>
#include <vector>
#include <cstring>
#include "sample_log.h"

bool file_exist(const std::string &path);

bool read_file(const std::string &path, std::vector<char> &data);
bool read_file(const std::string &path, char **data, size_t *len);
class MMap
{
private:
    void *_add = nullptr;
    int _size;

public:
    MMap() {}
    MMap(const char *file)
    {
        open_file(file);
    }
    ~MMap()
    {
        close_file();
    }

    bool open_file(const char *file)
    {
        _add = _mmap(file, &_size);
        if (!_add)
        {
            return false;
        }
        return true;
    }

    void close_file()
    {
        if (_add)
        {
            munmap(_add, _size);
            _add = nullptr;
            _size = 0;
        }
    }

    size_t size()
    {
        return _size;
    }

    void *data()
    {
        return _add;
    }

    static void *_mmap(const char *model_file, int *model_size)
    {
        auto *file_fp = fopen(model_file, "r");
        if (!file_fp)
        {

            return nullptr;
        }
        fseek(file_fp, 0, SEEK_END);
        *model_size = ftell(file_fp);
        fclose(file_fp);
        int fd = open(model_file, O_RDONLY);
        if (fd < 0)
        {
            fprintf(stderr, "[MMap] open failed for file %s: %s\n", model_file, strerror(errno));
            return nullptr;
        }

        void *mmap_addr = mmap(NULL, *model_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_addr == MAP_FAILED)
        {
            fprintf(stderr, "[MMap] mmap failed for file %s: %s\n", model_file, strerror(errno));
            close(fd);
            return nullptr;
        }
        close(fd);
        return mmap_addr;
    }
};