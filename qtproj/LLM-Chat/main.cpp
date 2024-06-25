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

        cmd.add<std::string>("data_en", 0, "", false);
        cmd.add<std::string>("data_zn", 0, "", false);

        cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
        cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
        cmd.add<int>("tokenizer_type", 0, "tokenizer type 0:LLaMa 1:Qwen", false, attr.tokenizer_type);
        cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
        cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

        cmd.add<std::string>("template_prefill_filename_axmodel", 0, "axmodel path template", true, attr.template_prefill_filename_axmodel);
        cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
        cmd.add<std::string>("filename_vpm_resampler_axmodedl", 0, "vpm resampler axmodel path", false, attr.filename_vpm_resampler_axmodedl);
        cmd.add<int>("vpm_width", 0, "vpm width", false, attr.vpm_width);
        cmd.add<int>("vpm_height", 0, "vpm height", false, attr.vpm_height);

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

        attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
        attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");
        attr.filename_vpm_resampler_axmodedl = cmd.get<std::string>("filename_vpm_resampler_axmodedl");
        attr.vpm_width = cmd.get<int>("vpm_width");
        attr.vpm_height = cmd.get<int>("vpm_height");

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

        std::vector<char> tmp_data;
        read_file(cmd.get<std::string>("data_en"), tmp_data);
        std::vector<unsigned short> en_data(tmp_data.size() / 2);
        memcpy(en_data.data(), tmp_data.data(), tmp_data.size());

        read_file(cmd.get<std::string>("data_zn"), tmp_data);
        std::vector<unsigned short> zn_data(tmp_data.size() / 2);
        memcpy(zn_data.data(), tmp_data.data(), tmp_data.size());

        w.m_data_en = en_data;
        w.m_data_zh = zn_data;
    }

    w.show();
    return a.exec();
}
