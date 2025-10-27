#include "mrope_qwen3.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

// 打印三维位置ID函数
void print_position_ids(const std::vector<std::vector<int>>& position_ids) {
    // 遍历三个维度（temporal, height, width）
    for (int dim = 0; dim < 3; ++dim) {
        std::cout << "=== Dimension " << dim << " ==="<< position_ids[dim].size() << std::endl;
        
        // 遍历每个batch
        // for (size_t batch_idx = 0; batch_idx < position_ids[dim].size(); ++batch_idx) {
        //     std::cout << "Batch " << batch_idx << ": ";
            
            // 遍历序列中的每个token
            for (int val : position_ids[dim]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        // }
    }
}


int main(int argc, char *argv[]){
    
    Config config;
    config.vision_config.spatial_merge_size = 2;
    config.image_token_id = 151655;
    config.video_token_id = 151656;
    config.vision_start_token_id = 151652;
    config.vision_config.tokens_per_second = 2;

    std::vector<std::vector<int>> POSITION_IDS={{},{},{}};

    std::vector<int> input_ids;
    readtxt("input_ids.txt", input_ids);

    

    std::vector<int> position_ids_gt;
    readtxt("position_ids.txt", position_ids_gt);

    int len = input_ids.size();
    POSITION_IDS[0].insert(POSITION_IDS[0].end(), position_ids_gt.begin(), position_ids_gt.begin()+len);
    POSITION_IDS[1].insert(POSITION_IDS[1].end(), position_ids_gt.begin()+len, position_ids_gt.begin()+len*2);
    POSITION_IDS[2].insert(POSITION_IDS[2].end(), position_ids_gt.begin()+len*2, position_ids_gt.begin()+len*3);

    std::vector<std::vector<int>> image_grid_thw={{1,  86, 128}};
    std::vector<std::vector<int>> video_grid_thw;
  

    auto position_ids = get_rope_index_qwen3(config, input_ids, image_grid_thw, video_grid_thw);

    print_position_ids(position_ids);

    for(int i=0;i<POSITION_IDS.size();i++){
        for(int j=0; j<POSITION_IDS[i].size();j++){
            if(POSITION_IDS[i][j]!=position_ids[i][j]){
                std::cout<<"check failed"<<std::endl;
                break;
            }
        }
    }

    std::cout<<"check success"<<std::endl;
}