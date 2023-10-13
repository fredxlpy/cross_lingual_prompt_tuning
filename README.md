# Cross-Lingual Prompt Tuning

This repository contains the code for **Soft Prompt Tuning for Cross-Lingual Transfer: When Less is More**.

All requirements to run the code can be found in `requirements.txt` and all required packages can be installed with `pip install -r requirements.txt`.

## Example
To train, simply run the following command:
```
python src/run_training.py \
    --source_lang en \
    --n_val_samples_per_class 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --lr 5e-3 \
    --n_epochs 20 \
    --max_steps -1 \
    --logging_steps 1 \
    --max_seq_length 128 \
    --prompt_length 2 \
    --n_train_samples_per_class 8 \
    --reparameterization \
    --reparam_hidden_size 200 \
    --seed 0
```
where \
    `--source_lang` is the source language. \
    `--prompt_length` is the length of the soft prompt. \
    `--n_train_samples_per_class` s the number of training samples per class. \
    `--n_val_samples_per_class` is the umber of validation samples per class. \
    `--per_device_train_batch_size` is the batch size for training (per device). \
    `--per_device_eval_batch_size` is the batch size for evaluation (per device). \
    `--learning_rate` is the learning rate used for training. \
    `--n_epochs` is the umber of training epochs. \
    `--max_steps` is the maximum number of training steps. \
    `--logging_steps` is the number of steps between logging during training. \
    `--seed` is the random seed for reproducibility. \
    `--max_seq_length` is the maximum sequence length for tokenization. \
    `--output_dir` is the directory to save the model checkpoints. \
    `--reparameterization` specifies whether to use reparameterization for prompt embeddings. \
    `--reparam_hidden_size` is the hidden size for reparameterization network. \

To test the prompt, run the following:
```
python src/run_evaluation.py \
    --n_eval_samples_per_class -1 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 8 \
    --source_lang en \
    --n_train_samples_per_class 8 \
    --prompt_length 2 \
    --per_device_train_batch_size 8 \
    --lr 5e-3 \
    --target_lang ar bg de el en es fr hi ru sw th tr ur vi zh \
    --reparameterization \
    --seed 0
```