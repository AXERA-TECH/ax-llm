#include "Tokenizer.hpp"

#include "sentencepiece_processor.h"
#include "builtin_pb/sentencepiece.pb.h"

#include "QwenTokenizer.hpp"

// #include "chatglm.h"

#include "httplib.h"
#include "json.hpp"

#include "sample_log.h"
#include "string_utility.hpp"
#include "memory_utils.hpp"

class TokenizerLLaMa : public BaseTokenizer
{
protected:
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

    bool Encode(std::string input, std::vector<int> &output, bool b_img_prompt = false) override
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

    std::vector<int> Encode(std::string input, bool b_img_prompt = false) override
    {
        std::vector<int> output;
        Encode(input, output, b_img_prompt);
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

class TokenizerMINICPM : public TokenizerLLaMa
{
public:
    std::string Decode(const std::vector<int> input) override
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        return spt.text();
    }
};

class TokenizerPhi3 : public BaseTokenizer
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

    bool Encode(std::string input, std::vector<int> &output, bool b_img_prompt = false) override
    {
        auto ret = sp.Encode(input, &output);
        if (!ret.ok())
        {
            ALOGE("%s", ret.error_message());
            return false;
        }
        output.insert(output.begin(), 32010); //"<|user|>"
        output.push_back(32007);              //"<|end|>"
        output.push_back(32001);              //"<|assistant|>"
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

    std::vector<int> Encode(std::string input, bool b_img_prompt = false) override
    {
        std::vector<int> output;
        Encode(input, output, b_img_prompt);
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
        return 32007;
    }

    bool isEnd(int id) override
    {
        return id == GetEosID() || id > 31999;
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
            ALOGE("tokenizer model file(%s) not exist", model_path.c_str());
            return false;
        }

        sp.reset(new QwenTokenizer(model_path, QwenConfig()));

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return true;
    }

    bool Encode(std::string input, std::vector<int> &output, bool b_img_prompt = false) override
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

    std::vector<int> Encode(std::string input, bool b_img_prompt = false) override
    {
        std::vector<int> output;
        Encode(input, output, b_img_prompt);
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

// class TokenizerGLM3 : public BaseTokenizer
// {
//     std::shared_ptr<chatglm::ChatGLM3Tokenizer> sp;
//     bool _b_bos, _b_eos;

// private:
//     /* data */
// public:
//     bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
//     {
//         if (!file_exist(model_path))
//         {
//             ALOGE("tokenizer model file(%s) not exist", model_path.c_str());
//             return false;
//         }
//         // std::vector<char> sp_model_data;
//         // read_file(model_path, sp_model_data);
//         // std::string_view serialized_model_proto(sp_model_data.data(), sp_model_data.size());

//         sp.reset(new chatglm::ChatGLM3Tokenizer(model_path));

//         this->_b_bos = b_bos;
//         this->_b_eos = b_eos;
//         return true;
//     }

//     bool Encode(std::string input, std::vector<int> &output) override
//     {
//         if (_b_bos)
//         {
//             // input += "<|im_start|>";
//         }
//         if (_b_eos)
//         {
//             // input += "<|endoftext|>";
//         }
//         output = sp->encode(input, 1024);

//         return true;
//     }

//     std::vector<int> Encode(std::string input) override
//     {
//         std::vector<int> output;
//         Encode(input, output);
//         return output;
//     }

//     std::string Decode(const std::vector<int> input) override
//     {
//         return sp->decode(input);
//     }

//     int GetBosID() override
//     {
//         return sp->sp.bos_id();
//     }

//     int GetEosID() override
//     {
//         return sp->sp.eos_id();
//     }
// };

class Tokenizer_Http : public BaseTokenizer
{
    std::shared_ptr<httplib::Client> cli;
    bool _b_bos, _b_eos;

    std::string base_url;

    int bos_id, eos_id;

private:
    /* data */
public:
    bool Init(std::string model_path = "http://localhost:8080", bool b_bos = true, bool b_eos = false) override
    {
        base_url = model_path;
        try
        {
            cli = std::make_shared<httplib::Client>(base_url);
            cli->set_connection_timeout(1);
            cli->set_read_timeout(1);
            cli->set_write_timeout(1);
            {
                auto ret = cli->Get("/bos_id");
                auto rep = ret.value();
                if (rep.status != 200)
                {
                    ALOGE("get bos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                bos_id = j["bos_id"];
            }

            {
                auto ret = cli->Get("/eos_id");
                auto rep = ret.value();
                if (rep.status != 200)
                {
                    ALOGE("get eos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                eos_id = j["eos_id"];
            }
            printf("bos_id: %d, eos_id: %d\n", bos_id, eos_id);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return true;
    }

    bool Encode(std::string input, std::vector<int> &output, bool b_img_prompt = false) override
    {
        nlohmann::json j;
        j["text"] = input;
        j["img_prompt"] = b_img_prompt;
        auto ret = cli->Post("/encode", j.dump(), "application/json");
        auto rep = ret.value();
        if (rep.status != 200)
        {
            ALOGE("encode failed, status: %d", rep.status);
            return false;
        }
        nlohmann::json j2;
        try
        {
            j2 = nlohmann::json::parse(rep.body);
        }
        catch (const std::exception &e)
        {
            ALOGE("json parse failed: %s", e.what());
            ALOGE("%s", rep.body.c_str());
            return false;
        }

        std::vector<int> out = j2["token_ids"];
        output = out;
        // output = sp->encode(input, 1024);
        if (_b_bos)
        {
            output.insert(output.begin(), bos_id);
        }
        if (_b_eos)
        {
            output.push_back(eos_id);
        }

        return true;
    }

    std::vector<int> Encode(std::string input, bool b_img_prompt = false) override
    {
        std::vector<int> output;
        Encode(input, output, b_img_prompt);
        return output;
    }

    std::string Decode(const std::vector<int> input) override
    {
        int cnt = 2;
        std::string out_str = "";
        while (cnt--)
        {
            nlohmann::json j;
            j["token_ids"] = input;
            auto ret = cli->Post("/decode", j.dump(), "application/json");
            auto rep = ret.value();
            if (rep.status != 200)
            {
                ALOGE("decode failed, status: %d, try again", rep.status);
                ALOGE("%s", rep.body.c_str());
                usleep(1000 * 1000);
                continue;
            }
            try
            {
                nlohmann::json j2 = nlohmann::json::parse(rep.body);
                out_str = j2["text"];
                break;
            }
            catch (const std::exception &e)
            {
                ALOGE("json parse failed: %s, try again", e.what());
                ALOGE("%s", rep.body.c_str());
                usleep(1000 * 1000);
                continue;
            }
        }
        return out_str;
    }

    int GetBosID() override
    {
        return bos_id;
    }

    int GetEosID() override
    {
        return eos_id;
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type)
{
    switch (type)
    {
    case TKT_LLaMa:
        return std::make_shared<TokenizerLLaMa>();
    case TKT_MINICPM:
        return std::make_shared<TokenizerMINICPM>();
    case TKT_HTTP:
        return std::make_shared<Tokenizer_Http>();
    case TKT_Qwen:
        return std::make_shared<TokenizerQwen>();
    case TKT_Phi3:
        return std::make_shared<TokenizerPhi3>();
    default:
        return nullptr;
    }
}