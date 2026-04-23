#pragma once
#include <stdio.h>
#include <cerrno>
#include <fstream>
#include <vector>
#include <string>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

bool file_exist(const std::string &path);

bool read_file(const std::string &path, std::vector<char> &data);
bool read_file(const std::string &path, char **data, size_t *len);
class MMap
{
private:
    void *_add = nullptr;
    size_t _size = 0;
#ifdef _WIN32
    HANDLE _file_handle = INVALID_HANDLE_VALUE;
    HANDLE _mapping_handle = nullptr;
#endif

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
#ifdef _WIN32
        close_file();

        _file_handle = CreateFileA(file, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (_file_handle == INVALID_HANDLE_VALUE)
        {
            return false;
        }

        LARGE_INTEGER file_size = {};
        if (!GetFileSizeEx(_file_handle, &file_size) || file_size.QuadPart <= 0)
        {
            close_file();
            return false;
        }
        _size = static_cast<size_t>(file_size.QuadPart);

        _mapping_handle = CreateFileMappingA(_file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!_mapping_handle)
        {
            close_file();
            return false;
        }

        _add = MapViewOfFile(_mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (!_add)
        {
            close_file();
            return false;
        }
#else
        _add = _mmap(file, &_size);
        if (!_add)
        {
            return false;
        }
#endif
        return true;
    }

    void close_file()
    {
        if (_add)
        {
#ifdef _WIN32
            UnmapViewOfFile(_add);
#else
            munmap(_add, _size);
#endif
            _add = nullptr;
        }
#ifdef _WIN32
        if (_mapping_handle)
        {
            CloseHandle(_mapping_handle);
            _mapping_handle = nullptr;
        }
        if (_file_handle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(_file_handle);
            _file_handle = INVALID_HANDLE_VALUE;
        }
#endif
        _size = 0;
    }

    size_t size() const
    {
        return _size;
    }

    void *data()
    {
        return _add;
    }

    const void *data() const
    {
        return _add;
    }

#ifndef _WIN32
    static void *_mmap(const char *model_file, size_t *model_size)
    {
        auto *file_fp = fopen(model_file, "rb");
        if (!file_fp)
        {

            return nullptr;
        }
        fseek(file_fp, 0, SEEK_END);
        *model_size = static_cast<size_t>(ftell(file_fp));
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
#endif
};
