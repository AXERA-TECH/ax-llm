./main \
--template_filename_axmodel tinyllama-int8-01/tinyllama_l%d.axmodel \
--axmodel_num 22 \
--filename_post_axmodel tinyllama-int8-01/tinyllama_post.axmodel \
--eos 0 \
--live_print 1 \
--continue 1 \
--prompt "$1"
