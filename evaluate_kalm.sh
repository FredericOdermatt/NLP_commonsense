cd "$(dirname "$0")"

MODEL_DIR=$1
fairseq-interactive \
    --path $1 ./KaLM/preprocess_taskC_data/subtaskC_data-bin \
    --beam 5 --source-lang source --target-lang target \
    --tokenizer moses \
    --bpe gpt2 --gpt2-encoder-json ./KaLM/preprocess_taskC_data/encoder.json --gpt2-vocab-bpe ./KaLM/preprocess_taskC_data/vocab.bpe
