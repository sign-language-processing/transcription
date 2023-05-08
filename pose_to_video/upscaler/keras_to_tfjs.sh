rm -rf training/web_model
rm -f training/web_model.zip
tensorflowjs_converter --input_format=keras training/model.h5 training/web_model
zip training/web_model.zip training/web_model/*