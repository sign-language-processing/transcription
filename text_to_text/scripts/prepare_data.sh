python -m text_to_text.prepare_data \
    --source_file 'text_to_text/data/data.csv.gz' \
    --target_file 'text_to_text/data/train.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --end_percent 90

python -m text_to_text.prepare_data \
    --source_file 'text_to_text/data/data.csv.gz' \
    --target_file 'text_to_text/data/validation.json' \
    --index_col 0 \
    --wrapper_col 'translation' \
    --start_percent 90