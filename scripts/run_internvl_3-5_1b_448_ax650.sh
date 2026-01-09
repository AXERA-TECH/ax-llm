AXMODEL_DIR=./internvl3-5_axmodel/

./main \
--template_filename_axmodel "${AXMODEL_DIR}qwen3_p128_l%d_together.axmodel" \
--axmodel_num 28 \
--filename_image_encoder_axmodedl "./vit-models/internvl_vit_model_1x448x448x3.axmodel" \
--bos 0 --eos 0 \
--dynamic_load_axmodel_layer 0 \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "internvl3-5-1b_tokenizer.txt" \
--filename_post_axmodel "${AXMODEL_DIR}/qwen3_post.axmodel" \
--use_topk 0 \
--filename_tokens_embed "${AXMODEL_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 151936 \
--tokens_embed_size 1024 \
--patch_size 14 \
--use_mrope 0 \
--temporal_patch_size 1 \
--live_print 1 \
--continue 1 \
--video 0 \
--img_width 448 \
--img_height 448 \
--vision_start_token_id 151652 \
--use_mrope 0 \
--post_config_path post_config.json 
