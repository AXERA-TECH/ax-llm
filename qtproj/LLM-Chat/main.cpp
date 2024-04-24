#include "mainwindow.h"
#include "style/DarkStyle.h"
#include <QApplication>
#include "src/runner/LLM.hpp"
#include "src/cmdline.hpp"

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);
    MainWindow w;

    {
        LLMAttrType attr;
        attr.template_filename_axmodel = "tinyllama-bf16/tinyllama_l%d.axmodel";
        attr.filename_post_axmodel = "tinyllama-bf16/tinyllama_post.axmodel";
        attr.filename_tokenizer_model = "tokenizer.model";
        attr.filename_tokens_embed = "tinyllama-bf16/tinyllama.model.embed_tokens.weight.bfloat16.bin";


        cmdline::parser cmd;
        cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
        cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
        cmd.add<int>("tokenizer_type", 0, "tokenizer type 0:LLaMa 1:Qwen", false, attr.tokenizer_type);
        cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
        cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

        cmd.add<bool>("bos", 0, "", false, attr.b_bos);
        cmd.add<bool>("eos", 0, "", false, attr.b_eos);
        cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
        cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
        cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

        cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);
        cmd.add<bool>("dynamic_load_axmodel_layer", 0, "it can save cmm memory", false, attr.b_dynamic_load_axmodel_layer);

        cmd.parse_check(argc, argv);

        attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
        attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
        attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
        attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
        attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
        attr.b_bos = cmd.get<bool>("bos");
        attr.b_eos = cmd.get<bool>("eos");
        attr.axmodel_num = cmd.get<int>("axmodel_num");
        attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
        attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

        attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");
        attr.b_dynamic_load_axmodel_layer = cmd.get<bool>("dynamic_load_axmodel_layer");

        attr.runing_callback = llm_running_callback;
        attr.reserve = &w;
        if (!w.InitLLM(attr))
        {
            return -1;
        }
    }

    w.show();
    return a.exec();
}
