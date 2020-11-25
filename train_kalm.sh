# specify where to save checkpoints and final model
SAVE_DIR=$SCRATCH/KaLM/trained_models/newtest

cd "$(dirname "$0")"

cat /proc/driver/nvidia/version

TOTAL_NUM_UPDATES=200000  #1200
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=100     #1600
UPDATE_FREQ=1
BART_PATH=./KaLM/bart.large/model.pt
echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES

fairseq-train ./KaLM/preprocess_taskC_data/subtaskC_data-bin \
    --restore-file $BART_PATH \
    --save-dir $SAVE_DIR \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --num-workers 10 \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --disable-validation \
    --log-format simple

#--no-epoch-checkpoints \
