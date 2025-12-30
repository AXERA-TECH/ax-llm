AXMODEL_DIR=./Qwen3-VL-2B-Instruct-AX650-c128_p1152-int4/

./main \
--template_filename_axmodel "${AXMODEL_DIR}/qwen3_vl_text_p128_l%d_together.axmodel" \
--axmodel_num 28 \
--filename_image_encoder_axmodedl "${AXMODEL_DIR}/Qwen3-VL-2B-Instruct_vision.axmodel" \
--bos 0 --eos 0 \
--dynamic_load_axmodel_layer 0 \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "qwen3_tokenizer.txt" \
--filename_post_axmodel "${AXMODEL_DIR}/qwen3_vl_text_post.axmodel" \
--use_topk 0 \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 151936 \
--tokens_embed_size 2048 \
--patch_size 16 \
--live_print 1 \
--continue 1 \
--video 1 \
--img_width 384 \
--img_height 384 \
--vision_start_token_id 151652 \
--post_config_path post_config.json 
