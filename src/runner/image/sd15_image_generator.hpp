#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openai_api/types.hpp"

namespace sd15 {

struct ImageModelVariant {
    std::string model_id;
    std::string model_dir;
    std::string openai_size;
    int width = 0;
    int height = 0;
    bool supports_img2img = false;
};

class ImageGenerator {
public:
    virtual ~ImageGenerator() = default;

    virtual bool init(const std::string& model_root, std::string& err) = 0;
    virtual std::vector<ImageModelVariant> variants() const = 0;
    virtual bool generate(const openai_api::ImageGenRequest& req,
                          std::vector<std::vector<uint8_t>>& png_images,
                          std::string& err) = 0;
};

std::unique_ptr<ImageGenerator> create_image_generator();

} // namespace sd15
