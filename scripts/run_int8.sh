./main_llama \
--template_filename_axmodel tinyllama-int8-01/tinyllama_l%d.axmodel \
--axmodel_num 22 \
--filename_post_axmodel tinyllama-int8-01/tinyllama_post.axmodel \
--max_token_len 512 --eos 0 \
--live_print 1 \
--prompt "$1"
