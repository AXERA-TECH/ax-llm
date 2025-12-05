# SmolVLM2-500M
AXMODEL_DIR=../../../SmolVLM2-500M-Video-Instruct.axera/python/SmolVLM2-500M-Video-Instruct_20151203_axmodel/
VIT=../../../SmolVLM2-500M-Video-Instruct.axera/python/vit-models/vision_model_1x3x512x512_NHwC_U8.axmodel
LAYER_NUM=32
EMBED_SIZE=960

# SmolVLM2-256M
AXMODEL_DIR=../../../SmolVLM2-500M-Video-Instruct.axera/python/SmolVLM2-256M-Video-Instruct_axmodel
VIT=../../../SmolVLM2-500M-Video-Instruct.axera/python/vit-models/vision_model_1x3x512x512_256M_NHwC_U8.axmodel
LAYER_NUM=30
EMBED_SIZE=576

./build/install/bin/main \
--template_filename_axmodel "${AXMODEL_DIR}/llama_p128_l%d_together.axmodel" \
--axmodel_num $LAYER_NUM \
--filename_image_encoder_axmodedl $VIT \
--bos 0 --eos 0 \
--dynamic_load_axmodel_layer 0 \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "http://127.0.0.1:8080" \
--filename_post_axmodel "${AXMODEL_DIR}/llama_post.axmodel" \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 49280 \
--tokens_embed_size $EMBED_SIZE \
--live_print 1 \
--continue 1 \
--video 0 \
--post_config_path post_config1.json 
