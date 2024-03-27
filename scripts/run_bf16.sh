./main \
--template_filename_axmodel tinyllama-bf16/tinyllama_l%d.axmodel \
--axmodel_num 22 \
--filename_post_axmodel tinyllama-bf16/tinyllama_post.axmodel \
--eos 0 \
--dynamic_load_axmodel_layer 0 \
--live_print 1 \
--prompt "$1"
