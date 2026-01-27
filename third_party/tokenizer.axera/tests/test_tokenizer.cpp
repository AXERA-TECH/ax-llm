#include "cmdline.hpp"
#include "timer.hpp"
#include "magic_enum.hpp"
#include "../include/BaseTokenizer.hpp"
#include <fstream>
#include <iomanip>

// ids1: {1,2,3}
// ids2: {1,2,3,4,5}
// diff_ids: {4,5}
// 这是用来比较两个 token id 序列的差异的函数，当新一轮对话用户输入prompt的时候，编码出来的token id 序列与上一轮的差异，
// 得到新增的token id 序列，这样历史对话的token id 序列就不需要重复计算kvcached了
std::vector<int> diff_token_ids(std::vector<int> ids1, std::vector<int> ids2)
{
    if (ids1.size() >= ids2.size())
    {
        return {};
    }
    for (int i = 0; i < ids1.size(); i++)
    {
        if (ids1[i] != ids2[i])
        {
            return {};
        }
    }
    std::vector<int> diff_ids(ids2.begin() + ids1.size(), ids2.end());
    return diff_ids;
}

void test_text_tokenizer(std::shared_ptr<BaseTokenizer> tokenizer)
{
    std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant";
    std::ifstream system_prompt_file("system_prompt.md");
    if (system_prompt_file.is_open())
    {
        std::string line;
        while (std::getline(system_prompt_file, line))
        {
            system_prompt += line;
        }
        system_prompt_file.close();
    }
    // 保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(true);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, system_prompt},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};
        timer t_cost;
        t_cost.start();
        auto chat_template = tokenizer->apply_chat_template(contents);
        t_cost.stop();
        printf("chat_template cost: %f ms\n", t_cost.cost());

        t_cost.start();
        std::vector<int> ids = tokenizer->encode(chat_template);
        t_cost.stop();
        printf("encode cost: %f ms\n", t_cost.cost());

        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }

    // 不保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(false);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, system_prompt},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }

    // 不保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(false);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, system_prompt},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "</think>\n\n你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }
}

void test_image_tokenizer(std::shared_ptr<BaseTokenizer> tokenizer)
{
    std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant";
    std::ifstream system_prompt_file("system_prompt.md");
    if (system_prompt_file.is_open())
    {
        std::string line;
        while (std::getline(system_prompt_file, line))
        {
            system_prompt += line;
        }
        system_prompt_file.close();
    }
    {
        std::vector<Content> contents = {
            {SYSTEM, TEXT, system_prompt},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, IMAGE, "帮我看看这张图片", 1, 256});

        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());
    }

    {
        std::vector<Content> contents = {
            {SYSTEM, TEXT, system_prompt},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, VIDEO, "帮我看看这个视频", 8, 256});
        timer t_cost;
        t_cost.start();
        auto ids2 = tokenizer->encode(contents);
        t_cost.stop();
        printf("encode cost: %f ms\n", t_cost.cost());
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());
    }
}

int main(int argc, char *argv[])
{
    std::stringstream model_type_str;
    model_type_str << "model type: \033[32m";
    for (auto name : magic_enum::enum_names<ModelType>())
    {
        model_type_str << std::setw(10) << name << " :" << magic_enum::enum_cast<ModelType>(name).value() << " ";
    }
    model_type_str << "\033[0m";

    std::string tokenizer_path = "../tests/assets/qwen3_tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.add<int>("model_type", 'm', model_type_str.str(), true);

    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");
    auto model_type = magic_enum::enum_cast<ModelType>(a.get<int>("model_type"));
    if (!model_type.has_value())
    {
        fprintf(stderr, "model type %d not found, please check model type in %s\n", a.get<int>("model_type"), model_type_str.str().c_str());
        return -1;
    }

    std::shared_ptr<BaseTokenizer> tokenizer = create_tokenizer((ModelType)model_type.value());
    if (!tokenizer->load(tokenizer_path))
    {
        fprintf(stderr, "load tokenizer failed");
        return -1;
    }

    test_text_tokenizer(tokenizer);
    test_image_tokenizer(tokenizer);
    return 0;
}