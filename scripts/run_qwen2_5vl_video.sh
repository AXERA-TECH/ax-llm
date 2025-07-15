AXMODEL_DIR=../Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512

./main \
--template_filename_axmodel "${AXMODEL_DIR}/qwen2_5_vl_p128_l%d_together.axmodel" \
--axmodel_num 36 \
--filename_image_encoder_axmodedl "${AXMODEL_DIR}/Qwen2.5-VL-3B-Instruct_vision_nhwc.axmodel" \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "http://127.0.0.1:8080" \
--filename_post_axmodel "${AXMODEL_DIR}/qwen2_5_vl_post.axmodel" \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 151936 \
--tokens_embed_size 2048 \
--live_print 1 \
--img_width 308 \
--img_height 308 \
--vision_start_token_id 151652 \
--post_config_path post_config.json \
--devices 0,1,2,3,4,5,6
