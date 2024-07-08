#pragma once
#include <string.h>
#include <string>
#include <vector>
#include <fstream>

#include "sample_log.h"

#include "memory_utils.hpp"

class LLaMaEmbedSelector
{
    MMap _embed_map;
    std::vector<unsigned short> _embeds;
    unsigned int _token_num, _embed_size;
    bool _use_mmap = false;

public:
    bool Init(std::string embed_path, unsigned int token_num, unsigned int embed_size, bool use_mmap = false)
    {
        _token_num = token_num;
        _embed_size = embed_size;
        _use_mmap = use_mmap;
        if (use_mmap)
        {
            ALOGI("LLaMaEmbedSelector use mmap");
            if (!_embed_map.open_file(embed_path.c_str()))
            {
                ALOGE("embed file(%s) open failed", embed_path.c_str());
                return false;
            }
        }
        else
        {
            std::ifstream fin(embed_path);
            if (!fin.is_open())
            {
                ALOGE("embed file(%s) open failed", embed_path.c_str());
                return false;
            }

            // get file size
            fin.seekg(0, std::ios::end);
            int file_size = fin.tellg();
            fin.seekg(0, std::ios::beg);
            if (file_size != token_num * embed_size * 2)
            {
                ALOGE("embed file(%s) size(%d) not equal token_num(%d) * embed_size(%d) * 2", embed_path.c_str(), file_size, token_num, embed_size);
                return false;
            }

            _embeds.resize(token_num * embed_size);
            fin.read((char *)_embeds.data(), file_size);
            fin.close();
        }

        return true;
    }

    void Deinit()
    {
        _embed_map.close_file();
        _embeds.clear();
    }

    void getByIndex(unsigned int index, std::vector<unsigned short> &embed)
    {
        if (index >= _token_num)
        {
            ALOGE("index(%d) > token_num(%d)", index, _token_num);
            return;
        }
        embed.resize(_embed_size);
        if (_use_mmap)
        {
            unsigned short *ptr = (unsigned short *)_embed_map.data();
            memcpy(embed.data(), ptr + index * _embed_size, _embed_size * sizeof(unsigned short));
        }
        else
        {
            memcpy(embed.data(), _embeds.data() + index * _embed_size, _embed_size * sizeof(unsigned short));
        }
    }

    void getByIndex(unsigned int index, unsigned short *embed)
    {
        if (index >= _token_num)
        {
            ALOGE("index(%d) > token_num(%d)", index, _token_num);
            return;
        }
        
        if (_use_mmap)
        {
            unsigned short *ptr = (unsigned short *)_embed_map.data();
            memcpy(embed, ptr + index * _embed_size, _embed_size * sizeof(unsigned short));
        }
        else
        {
            memcpy(embed, _embeds.data() + index * _embed_size, _embed_size * sizeof(unsigned short));
        }
    }

    std::vector<unsigned short> getByIndex(unsigned int index)
    {
        std::vector<unsigned short> embed;
        this->getByIndex(index, embed);
        return embed;
    }
};
