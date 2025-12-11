#include <vector>
#include <algorithm>
#include <optional>
#include <cassert>
#include "mrope.hpp"

#include <iostream>
#include <vector>
#include <numeric>  // std::iota

#include <vector>
#include <algorithm>
#include <limits>  // 用于std::numeric_limits
#include <stdexcept>  // 用于异常处理

int findMaxIn2DVector(const std::vector<std::vector<int>>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("输入二维vector为空");  // 处理空vector <button class="citation-flag" data-index="7">
    }

    int max_value = std::numeric_limits<int>::min();  // 初始化为最小值 <button class="citation-flag" data-index="1">
    bool has_elements = false;

    for (const auto& subvec : vec) {
        if (!subvec.empty()) {
            has_elements = true;
            // 使用std::max_element获取子vector的最大值 <button class="citation-flag" data-index="3">
            int sub_max = *std::max_element(subvec.begin(), subvec.end());
            if (sub_max > max_value) {
                max_value = sub_max;
            }
        }
    }

    if (!has_elements) {
        throw std::invalid_argument("所有子vector均为空");  // 处理全空子vector <button class="citation-flag" data-index="7">
    }

    return max_value;
}

// 生成范围序列 [0, text_len-1]
std::vector<int> generateRange(int text_len, int start) {
    std::vector<int> range(text_len);
    std::iota(range.begin(), range.end(), start);  // 填充从0开始的序列 <button class="citation-flag" data-index="4">
    return range;
}

// 扩展为多行矩阵
std::vector<std::vector<int>> expandToMatrix(const std::vector<int>& range, int rows) {
    std::vector<std::vector<int>> matrix(rows, range);  // 每一行都是range的副本 <button class="citation-flag" data-index="4">
    return matrix;
}

// 生成多维索引
std::vector<std::vector<int>> generateIndices(int grid_t, int grid_h, int grid_w) {
    std::vector<std::vector<int>> indices(3, std::vector<int>(grid_t * grid_h * grid_w));

    int idx = 0;
    for (int t = 0; t < grid_t; ++t) {
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                indices[0][idx] = t;  // 时间索引
                indices[1][idx] = h;  // 高度索引
                indices[2][idx] = w;  // 宽度索引
                ++idx;
            }
        }
    }

    return indices;
}

std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const std::vector<double>& second_per_grid_ts) 
{
    const int spatial_merge_size = config.vision_config.spatial_merge_size;
    const int image_token_id = config.image_token_id;
    const int video_token_id = config.video_token_id;
    const int vision_start_token_id = config.vision_start_token_id;
    
    std::vector<std::vector<int>> position_ids(3);
    std::vector<int> mrope_position_deltas;

    // 处理纯文本情况
    if (input_ids.empty() || (image_grid_thw.empty() && video_grid_thw.empty())) {
        // for (size_t b = 0; b < input_ids.size(); ++b) {
            int b=0;
            for (int i = 0; i < 3; ++i) {
                std::vector<int> seq(input_ids.size());
                // 手动实现递增序列（替代std::iota）
                for (size_t j = 0; j < seq.size(); ++j) {
                    seq[j] = j;
                }
                // position_ids[i].push_back(seq);
                position_ids[i].insert(position_ids[i].end(), seq.begin(),seq.end());
            }

            mrope_position_deltas.push_back(0);
        // }
        // return {position_ids, mrope_position_deltas};
        return position_ids;
    }

    // 处理多模态情况
    // for (size_t batch_idx = 0; batch_idx < input_ids.size(); ++batch_idx) {
        // const auto& ids = input_ids[batch_idx];
        const auto & ids = input_ids;
        // const auto& mask = attention_mask.empty() ? std::vector<int>(ids.size(), 1) : attention_mask[batch_idx];
        const auto mask = std::vector<int>(ids.size(), 1);
        
        // 过滤有效token
        std::vector<int> filtered_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (mask[i]) filtered_ids.push_back(ids[i]);
        }
    
        
        int image_nums = 0, video_nums = 0;

        for (size_t i = 0; i < filtered_ids.size()-1; ++i){
            if (filtered_ids[i] == vision_start_token_id) {
                if(filtered_ids[i+1]==config.image_token_id){
                    image_nums++;
                }
                if(filtered_ids[i+1]==config.video_token_id){
                    video_nums++;
                }
            }
        }

        int image_index = 0, video_index = 0;
        int ed_image = 0, ed_video = 0 ;
        std::vector<std::vector<int>> batch_pos(3);
        int st = 0;
        int remain_images = image_nums;
        int remain_videos = video_nums;
        std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
        // for (const auto& start_idx : vision_start_indices) {
        
        for(size_t i_=0; i_<image_nums+video_nums; ++i_){
            
            if(remain_images>0){
                for(size_t j=st; j<filtered_ids.size(); ++j){
                    if(filtered_ids[j]==config.image_token_id){
                        ed_image = j;
                        break;
                    }
                }
            }
            else{
                ed_image = filtered_ids.size()+1;
            }

            if(remain_videos>0){
                for(size_t j=st; j<filtered_ids.size(); ++j){
                    if(filtered_ids[j]==config.video_token_id){
                        ed_video = j;
                        break;
                    }
                }
            }
            else{
                ed_video = filtered_ids.size()+1;
            }

            int t,h,w;
            double second_per_grid_t;
            int ed;

            if(ed_image < ed_video){
                t = image_grid_thw[image_index][0];
                h = image_grid_thw[image_index][1];
                w = image_grid_thw[image_index][2];

                second_per_grid_t = 0;
                image_index += 1;
                remain_images -= 1;
                ed = ed_image;
            }
            else{
                t = video_grid_thw[video_index][0];
                h = video_grid_thw[video_index][1];
                w = video_grid_thw[video_index][2];
                
                if(!second_per_grid_ts.empty()){
                    second_per_grid_t = second_per_grid_ts[video_index];
                }
                else{
                    second_per_grid_t = 1.0;
                }

                video_index += 1;
                remain_videos -= 1;
                ed = ed_video;
            }

            int llm_grid_t, llm_grid_h, llm_grid_w;
            llm_grid_t = t;
            llm_grid_h = h / spatial_merge_size;
            llm_grid_w = w / spatial_merge_size;

            int text_len = ed -st;

            int st_idx;
            if(llm_pos_ids_list.empty()){
                st_idx = 0;
            }else{
                // auto max_iter = std::max_element(llm_pos_ids_list.back()[0].begin(), llm_pos_ids_list.back()[0].end()); 
                // st_idx = *max_iter + 1;

                st_idx = findMaxIn2DVector(llm_pos_ids_list.back());
            }
            auto range = generateRange(text_len, st_idx);
            auto expanded_matrix = expandToMatrix(range, 3);

            llm_pos_ids_list.push_back(expanded_matrix);
            std::vector<int> t_index;


            for(size_t ti=0; ti<llm_grid_t; ti++){
                for(size_t hw=0; hw<llm_grid_h*llm_grid_w; hw++){
                    t_index.push_back( ti*second_per_grid_t*config.vision_config.tokens_per_second + text_len + st_idx );
                }
            }

            std::vector<int> h_index;
            for(size_t ti=0; ti<llm_grid_t;ti++){
                for(size_t hi=0; hi<llm_grid_h; hi++){
                    for(size_t wi=0; wi<llm_grid_w; wi++){
                        h_index.push_back(hi + text_len + st_idx);
                    }
                }
            } 

            
            std::vector<int> w_index;
            for(size_t ti=0; ti<llm_grid_t;ti++){
                for(size_t hi=0; hi<llm_grid_h;hi++){
                    for(size_t wi=0; wi<llm_grid_w; wi++){
                        w_index.push_back(wi + text_len + st_idx);
                    }
                }
            }

            std::vector<std::vector<int>> thw_idx;
            thw_idx.push_back(t_index);
            thw_idx.push_back(h_index);
            thw_idx.push_back(w_index);
            llm_pos_ids_list.push_back(thw_idx);

            st = ed + llm_grid_t*llm_grid_h*llm_grid_w;

            if(st<filtered_ids.size()){
                if(llm_pos_ids_list.empty()){
                    st_idx = 0;
                }else{
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }

                text_len = filtered_ids.size() - st;

                auto range = generateRange(text_len, st_idx);
                auto expanded_matrix = expandToMatrix(range, 3);
                llm_pos_ids_list.push_back(expanded_matrix);
            }

            for(auto & item : llm_pos_ids_list){
                for(size_t pi=0; pi<position_ids.size();pi++){
                    position_ids[pi].insert(position_ids[pi].end(), item[pi].begin(), item[pi].end());   
                }
            }
        }
    // }
    
    return position_ids;
}