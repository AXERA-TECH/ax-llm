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
        _add = _safe_mmap(file, &_size);
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
        int fd = open(model_file, O_RDWR, 0644);
        void *mmap_add = mmap(NULL, *model_size, PROT_WRITE, MAP_SHARED, fd, 0);
        return mmap_add;
    }

    static void *_safe_mmap(const char *model_file, int *model_size)
    {
        struct stat st;
        if (stat(model_file, &st) != 0)
        {
            fprintf(stderr, "[MMap] stat failed for file %s: %s\n", model_file, strerror(errno));
            return nullptr;
        }

        *model_size = st.st_size;

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

        // 检测是否在 NFS（可选）
        char fs_type[64] = {};
        FILE *mnt = setmntent("/proc/mounts", "r");
        if (mnt)
        {
            struct mntent *ent;
            while ((ent = getmntent(mnt)) != nullptr)
            {
                if (strstr(model_file, ent->mnt_dir) == model_file)
                {
                    snprintf(fs_type, sizeof(fs_type), "%s", ent->mnt_type);
                    break;
                }
            }
            endmntent(mnt);
        }

        if (strcmp(fs_type, "nfs") == 0 || strcmp(fs_type, "nfs4") == 0)
        {
           ALOGW("[MMap][Warning] Using mmap on NFS may cause instability!");
        }

        close(fd);
        return mmap_addr;
    }
};