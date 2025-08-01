AXMODEL_DIR=../Qwen2.5-VL-7B-Instruct-AX650-chunk_prefill_1280

./build/install/bin/main \
--template_filename_axmodel "${AXMODEL_DIR}/qwen2_5_vl_p128_l%d_together.axmodel" \
--axmodel_num 28 \
--filename_image_encoder_axmodedl "${AXMODEL_DIR}/Qwen2.5-VL-7B-Instruct_vision.axmodel" \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "http://127.0.0.1:8091" \
--filename_post_axmodel "${AXMODEL_DIR}/qwen2_5_vl_post.axmodel" \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 152064 \
--tokens_embed_size 3584 \
--live_print 1 \
--video 0 \
--img_width 448 \
--img_height 448 \
--vision_start_token_id 151652 \
--post_config_path post_config.json \
--devices 0,1,2,3,4,5,6,7


# What are these attractions? Please give their names in Chinese and English
# assets/attractions