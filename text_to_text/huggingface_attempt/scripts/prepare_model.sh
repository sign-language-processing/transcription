python -m text_to_text.prepare_model \
    --source_model 'microsoft/mdeberta-v3-base' \
    --target_model 'gpt2' \
    --final_name 'initial-mdeberta-to-gpt2' \
    --outputs_dir 'text_to_text/models' \
    --model_max_length 128

