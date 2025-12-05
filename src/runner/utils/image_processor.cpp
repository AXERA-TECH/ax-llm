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
        std::cout<<"read image"<<std::endl;
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

std::vector<cv::Mat> splitImageSafe(cv::Mat src, int rows, int cols) {
    std::vector<cv::Mat> subImages;
    
    cv::Size size(1024, 1024);

    if(src.cols!=1024 || src.rows!=1024){
        cv::Mat img_rs;
        cv::resize(src, img_rs, size, 0, 0, cv::INTER_CUBIC);
        src = img_rs;
    }

    int subHeight = src.rows / rows;
    int subWidth = src.cols / cols;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 计算ROI，确保不越界
            int x = j * subWidth;
            int y = i * subHeight;
            int width = (j == cols - 1) ? src.cols - x : subWidth;
            int height = (i == rows - 1) ? src.rows - y : subHeight;
            
            cv::Rect roi(x, y, width, height);
            
            // 检查ROI是否有效
            if (roi.x >= 0 && roi.y >= 0 && 
                roi.x + roi.width <= src.cols && 
                roi.y + roi.height <= src.rows) {
                
                cv::Mat subImage = src(roi).clone();
                cv::Size size(512, 512);
                if(subImage.cols!=512 || subImage.rows!=512){
                    cv::resize(subImage, subImage, size, 0, 0, cv::INTER_CUBIC);
                }
                
                subImages.push_back(subImage);
            }
        }
    }
    
    return subImages;
}

std::vector<unsigned char> hwc_to_chw(const std::vector<unsigned char>& hwc_data, 
                                         int height, int width, int channels) {
    assert(hwc_data.size() == height * width * channels);
    
    std::vector<unsigned char> chw_data(height * width * channels);
    
    const unsigned char* hwc_ptr = hwc_data.data();
    unsigned char* chw_ptr = chw_data.data();
    
    int hw_size = height * width;
    
    for (int c = 0; c < channels; ++c) {
        unsigned char* channel_ptr = chw_ptr + c * hw_size;
        
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int hwc_index = (h * width + w) * channels + c;
                channel_ptr[h * width + w] = hwc_ptr[hwc_index];
            }
        }
    }
    
    return chw_data;
}

int Smolvlm2ImageProcessor(std::vector<cv::Mat>& src, std::vector<std::vector<unsigned char>>& output)
{
    if(src.empty()){
        return 0;
    }

    std::vector<cv::Mat> resized;
    cv::Size size(512, 512);
    
    for(auto& img: src){
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        auto splited = splitImageSafe(img, 2, 2);
        resized.insert(resized.end(), splited.begin(), splited.end());
        if(img.cols!=512 || img.rows!=512){
            cv::resize(img, img, size, 0, 0, cv::INTER_CUBIC);
        }
        resized.push_back(img);
    }

    output.clear();
    for(auto& img: resized)
    {
        std::vector<unsigned char> imgdata;
        imgdata.resize( 512 * 512 * 3);
        memcpy(imgdata.data(), img.data, 512 * 512 * 3);
        output.push_back(hwc_to_chw(imgdata, 512, 512, 3));
        // output.push_back(imgdata);
    }
    
    return 0;

}

int Smolvlm2VideoProcessor(std::vector<cv::Mat>& src, std::vector<std::vector<unsigned char>>& output)
{
    if(src.empty()){
        return 0;
    }

    output.clear();
    cv::Size size(512, 512);
    for(auto& img: src)
    {
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        std::vector<unsigned char> imgdata;
        imgdata.resize( 512 * 512 * 3);

        if(img.cols!=512 || img.rows!=512){
            cv::resize(img, img, size, 0, 0, cv::INTER_CUBIC);
        }

        memcpy(imgdata.data(), img.data, 512 * 512 * 3);
        output.push_back(hwc_to_chw(imgdata, 512, 512, 3));
        // output.push_back(imgdata);
    }
    
    return 0;
}