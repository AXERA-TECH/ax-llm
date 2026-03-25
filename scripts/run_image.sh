AXMODEL_DIR=Qwen3.5-2B-AX650-C128-P1152-CTX2047
MODEL_CONFIG=$AXMODEL_DIR/config.json

./build_650/install/bin/main \
--template_filename_axmodel "${AXMODEL_DIR}/qwen3_5_text_p128_l%d_together.axmodel" \
--axmodel_num 24 \
--filename_image_encoder_axmodedl "${AXMODEL_DIR}/qwen3_5_vision.axmodel" \
--bos 0 --eos 0 \
--dynamic_load_axmodel_layer 0 \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "scripts/qwen3_5_tokenizer.txt" \
--model_config_path "${MODEL_CONFIG}" \
--filename_post_axmodel "${AXMODEL_DIR}/qwen3_5_text_post.axmodel" \
--use_topk 0 \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 248320 \
--tokens_embed_size 2048 \
--patch_size 16 \
--live_print 1 \
--continue 1 \
--video 0 \
--img_width 384 \
--img_height 384 \
--vision_start_token_id 248053 \
--post_config_path post_config.json 
