#include "Tokenizer.hpp"

#include "sentencepiece_processor.h"
#include "builtin_pb/sentencepiece.pb.h"

#include "Tokenizer/QwenTokenizer.hpp"

#include "sample_log.h"
#include "string_utility.hpp"
#include "memory_utils.hpp"

class TokenizerLLaMa : public BaseTokenizer
{
    sentencepiece::SentencePieceProcessor sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
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

    bool Encode(std::string input, std::vector<int> &output) override
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

    std::vector<int> Encode(std::string input) override
    {
        std::vector<int> output;
        Encode(input, output);
        return output;
    }

    std::string Decode(const std::vector<int> input) override
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        std::string out = spt.pieces()[0].piece();
        if (*(unsigned short *)out.data() == 38626)
        {
            return " " + spt.text();
        }
        else
        {
            return spt.text();
        }
    }

    int GetBosID() override
    {
        return sp.bos_id();
    }

    int GetEosID() override
    {
        return sp.eos_id();
    }
};

class TokenizerQwen : public BaseTokenizer
{
    std::shared_ptr<QwenTokenizer> sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
    {
        if (!file_exist(model_path))
        {
            return false;
        }

        sp.reset(new QwenTokenizer(model_path, QwenConfig()));

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return true;
    }

    bool Encode(std::string input, std::vector<int> &output) override
    {
        if (_b_bos)
        {
            // input += "<|im_start|>";
        }
        if (_b_eos)
        {
            input += "<|endoftext|>";
        }
        output = sp->encode(input, 1024);

        return true;
    }

    std::vector<int> Encode(std::string input) override
    {
        std::vector<int> output;
        Encode(input, output);
        return output;
    }

    std::string Decode(const std::vector<int> input) override
    {
        return sp->decode(input);
    }

    int GetBosID() override
    {
        return -1;
    }

    int GetEosID() override
    {
        return sp->eos_token_id;
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type)
{
    switch (type)
    {
    case TKT_LLaMa:
        return std::make_shared<TokenizerLLaMa>();
    case TKT_Qwen:
        return std::make_shared<TokenizerQwen>();
    default:
        return nullptr;
    }
}