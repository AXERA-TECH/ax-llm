#include "runner/LLaMaEmbedSelector.hpp"
#include "runner/bfloat16.hpp"

#include "runner/Tokenizer.hpp"

#include "runner/LLaMa.hpp"
#include "cmdline.hpp"

int main(int argc, char *argv[])
{
    LLaMaAttrType attr;
    std::string prompt = "Hi";

    cmdline::parser cmd;
    cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);
    cmd.add<int>("max_token_len", 0, "max token len", false, attr.max_token_len);
    cmd.add<int>("kv_cache_num", 0, "count of kv cache(axmodel kv_cache input dim-0)", false, attr.kv_cache_num);
    cmd.add<int>("kv_cache_size", 0, "len of kv cache(axmodel kv_cache input dim-1)", false, attr.kv_cache_size);

    cmd.parse_check(argc, argv);

    prompt = cmd.get<std::string>("prompt");
    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");
    attr.max_token_len = cmd.get<int>("max_token_len");
    attr.kv_cache_num = cmd.get<int>("kv_cache_num");
    attr.kv_cache_size = cmd.get<int>("kv_cache_size");

    LLaMa lLaMa;
    lLaMa.Init(attr);

    auto output = lLaMa.Run(prompt);
    printf("%s\n", output.c_str());
    return 0;
}