rm -r training/web_model
rm training/web_model.zip
tensorflowjs_converter --quantize_float16 --weight_shard_size_bytes 8388608 \
  --input_format=keras training/model.h5 training/web_model
zip training/web_model.zip training/web_model/*