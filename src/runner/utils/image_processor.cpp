#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "files.hpp"
#include "image_processor.hpp"
#include <iostream>

std::vector<cv::Mat> ReadImages(std::string path){
    std::vector<cv::Mat> src;

    if(is_file(path)){
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        src.push_back(img);
    }
    else if(is_directory(path)){
        auto paths = list_files(path);
        
        for(auto &p : paths){
            std::cout<<p<<std::endl;
            cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
            src.push_back(img);
        }
    }
    else{
        std::cerr << "错误的路径: " << path << std::endl;
    }

    return src;
}

std::pair<int, int> SmartResize(int height, int width, int factor){
    int h_bar = height/factor;
    int w_bar = width/factor;

    h_bar *= factor;
    w_bar *= factor;
    return {h_bar, w_bar};
}

void normalizeMeanStd(cv::Mat& image) {
    // 确保输入图像是浮点类型（避免整数溢出）
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);  // 转换为32位浮点格式 <button class="citation-flag" data-index="1">

    // 计算均值和标准差
    cv::Scalar mean, stddev;
    cv::meanStdDev(floatImage, mean, stddev);  // 计算均值和标准差 <button class="citation-flag" data-index="2">

    // 避免除以零：如果标准差为0，设置为一个小值（如1e-6）
    for (int i = 0; i < floatImage.channels(); ++i) {
        if (stddev[i] < 1e-6) {
            stddev[i] = 1e-6;
        }
    }

    // 归一化：减去均值并除以标准差
    floatImage -= mean;  // 减去均值 <button class="citation-flag" data-index="4">
    floatImage /= stddev;  // 除以标准差 <button class="citation-flag" data-index="5">

    // 将结果转换回原始数据类型（如8位无符号整数）
    floatImage.convertTo(image, image.type());  // 转换回原始格式 <button class="citation-flag" data-index="6">
}

int Qwen2VideoProcessor( std::vector<cv::Mat>& src, std::vector<std::vector<unsigned char>>& output, 
                            int tgt_h, int tgt_w,
                            int temporal_patch_size, int merge_size, int patch_size){

    if(src.empty()){
        return 0;
    }

    int height = src[0].rows;
    int width = src[0].cols;

    // auto [tgt_h, tgt_w] = SmartResize(height, width, 28);

    cv::Size size(tgt_w, tgt_h);
    std::vector<cv::Mat> imgs_resized;
    
    for(auto& img: src){
        cv::Mat img_rs;
        if(img.cols!=tgt_w || img.rows!=tgt_h){
            cv::resize(img, img_rs, size, 0, 0, cv::INTER_CUBIC);
        }else{
            img_rs = img;
        }
        
        cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);
        imgs_resized.push_back(img_rs);
    }
    
    if(imgs_resized.empty()){
        return 0;
    }

    if(imgs_resized.size()%2!=0){
        imgs_resized.push_back(imgs_resized.back());
    }

    std::vector<unsigned char> patches;
    patches.resize( imgs_resized.size()* tgt_w*tgt_h* 3);
    for(size_t i=0; i<imgs_resized.size(); ++i){
        memcpy(patches.data()+i*tgt_w*tgt_h*3, imgs_resized[i].data, tgt_w*tgt_h* 3);
    }

    int grid_t = imgs_resized.size() / temporal_patch_size;
    int channel = imgs_resized[0].channels();
    int grid_h = tgt_h/patch_size;
    int grid_w = tgt_w/patch_size;

    // channel = patches.shape[3]
    // patches = patches.reshape(
    //     grid_t,                     # 0
    //     self.temporal_patch_size,   # 1
    //     grid_h // self.merge_size,  # 2
    //     self.merge_size,            # 3
    //     self.patch_size,            # 4
    //     grid_w // self.merge_size,  # 5
    //     self.merge_size,            # 6
    //     self.patch_size,            # 7
    //     channel                     # 8
    // )   
    // patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7, 8 )

    for(size_t d0=0; d0<grid_t; d0++){
        std::vector<unsigned char> out_t;
        for(size_t d2=0; d2<grid_h/merge_size; d2++){
            for(size_t d5=0; d5<grid_w/merge_size; d5++){
                for(size_t d3=0; d3<merge_size; d3++ ){
                    for(size_t d6=0; d6<merge_size; d6++){
                        for(size_t d1=0; d1<temporal_patch_size; d1++){
                            for(size_t d4=0; d4<patch_size; d4++){
                                for(size_t d7=0; d7<patch_size; d7++){
                                    for(size_t d8=0; d8<channel; d8++){
                                        size_t idx = d0*temporal_patch_size*grid_h*patch_size*grid_w*patch_size*channel;
                                        idx += d1*grid_h*patch_size*grid_w*patch_size*channel;
                                        idx += d2*merge_size*patch_size*grid_w*patch_size*channel;
                                        idx += d3*patch_size*grid_w*patch_size*channel;
                                        idx += d4*grid_w*patch_size*channel;
                                        idx += d5*merge_size*patch_size*channel;
                                        idx += d6*patch_size*channel;
                                        idx += d7*channel;
                                        idx += d8;

                                        out_t.push_back(patches[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output.push_back(out_t);
    }

    // std::vector<size_t> ret={grid_t, grid_h*grid_w, temporal_patch_size*patch_size*patch_size, channel};
    // return ret;
    return 0;

}

