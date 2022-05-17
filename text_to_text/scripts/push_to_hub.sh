export WANDB_DISABLED="true"
export CUDA_VISIBLE_DEVICES=
python -m text_to_text.run_translation \
  --output_dir 'text_to_text/outputs' \
  --train_file 'text_to_text/data/train.json' \
  --validation_file 'text_to_text/data/validation.json' \
  --model_name_or_path 'text_to_text/models/initial-mdeberta-to-gpt2' \
  --source_lang 'src' \
  --target_lang 'tgt' \
  --push_to_hub \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0