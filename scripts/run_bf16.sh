./main_llama \
--template_filename_axmodel tinyllama-bf16/tinyllama_l%d.axmodel \
--axmodel_num 22 \
--filename_post_axmodel tinyllama-bf16/tinyllama_post.axmodel \
--max_token_len 512 --eos 0 \
--dynamic_load_axmodel_layer 1 \
--live_print 1 \
--prompt "$1"
