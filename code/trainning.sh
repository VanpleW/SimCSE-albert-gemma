"""
    Due to limited computational resources, 
    only one set of training parameter is used for each model,
    that might lead to non-ideal results for some models
"""


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

function llama2-7B() {
    poetry run python train.py \
        --model_name llama \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/llama
}

function gemma-7B() {
    poetry run python train.py \
        --model_name gemma \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/gemma
}

# call for training starts
bert-base-uncased