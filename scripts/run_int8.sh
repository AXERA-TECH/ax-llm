./main \
--template_filename_axmodel tinyllama-int8-01/tinyllama_l%d.axmodel \
--axmodel_num 22 \
--filename_post_axmodel tinyllama-int8-01/tinyllama_post.axmodel \
--filename_tokens_embed tinyllama-int8-01/tinyllama.model.embed_tokens.weight.bfloat16.bin \
--eos 0 \
--live_print 1 \
--continue 1 \
--prompt "$1"
