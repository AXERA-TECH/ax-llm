#pragma once
#include "sentencepiece_processor.h"
#include "builtin_pb/sentencepiece.pb.h"

#include "sample_log.h"

class Tokenizer
{
    sentencepiece::SentencePieceProcessor sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false)
    {
        auto ret = sp.Load(model_path);
        if (!ret.ok())
        {
            ALOGE("%s", ret.error_message());
            return false;
        }

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return ret.ok();
    }

    bool Encode(std::string input, std::vector<int> &output)
    {
        auto ret = sp.Encode(input, &output);
        if (!ret.ok())
        {
            ALOGE("%s", ret.error_message());
            return false;
        }
        if (_b_bos)
        {
            output.insert(output.begin(), sp.bos_id());
        }
        if (_b_eos)
        {
            output.push_back(sp.eos_id());
        }
        return true;
    }

    std::vector<int> Encode(std::string input)
    {
        std::vector<int> output;
        Encode(input, output);
        return output;
    }

    std::string Decode(const std::vector<int> input)
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        return spt.text();
    }

    int GetBosID()
    {
        return sp.bos_id();
    }

    int GetEosID()
    {
        return sp.eos_id();
    }
};
