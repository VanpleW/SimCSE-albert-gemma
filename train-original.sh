# original configurations described in the paper are used
# see: Table A.1 of Appendix.A


function bert-base-uncased() {
    poetry run python train.py \
        --model_name bert-base-uncased \
        --batch_size 64 \
        --lr 3e-5 \
        --output_dir ./outputs/bert-base-uncased
}

function bert-large-uncased() {
    poetry run python train.py \
        --model_name bert-large-uncased \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/bert-large-uncased
}


function roberta-base() {
    poetry run python train.py \
        --model_name roberta-base \
        --batch_size 512 \
        --lr 1e-5 \
        --output_dir ./outputs/roberta-base
}


function roberta-large() {
    poetry run python train.py \
        --model_name roberta-large \
        --batch_size 512 \
        --lr 3e-5 \
        --output_dir ./outputs/roberta-large
}

function albert-base-v2() {
    poetry run python train.py \
        --model_name albert-base-v2 \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/albert-base-v2
}

function llama() {
    poetry run python train.py \
        --model_name llama \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/llama
}

function gemma() {
    poetry run python train.py \
        --model_name gemma \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/gemma
}

# run training!
# you should change the function name to run different models
bert-base-uncased