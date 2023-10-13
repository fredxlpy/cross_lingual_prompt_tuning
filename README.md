# Cross-Lingual Prompt Tuning

This repository contains the code for **Soft Prompt Tuning for Cross-Lingual Transfer: When Less is More**.

All requirements to run the code can be found in `requirements.txt` and all required packages can be installed with `pip install -r requirements.txt`.

## Example
To train, simply run the following command:
```
python src/run_training.py \
    --source_lang en \
    --prompt_length 5 \
    --model_name xglm-564M \
    --n_train_samples_per_class 8 \
    --per_device_train_batch_size 8 \
    --lr 5e-3 \
    --n_epochs 20 \
    --max_steps -1 \
    --seed 0 \
    --reparameterization \
    --reparam_hidden_size 200 \
    --model_freezing
```
where \
    `--source_lang` is the source language. \
    `--prompt_length` is the length of the soft prompt. \
    `--model_name` is the name of the given model. \
    `--n_train_samples_per_class` s the number of training samples per class. \
    `--per_device_train_batch_size` is the batch size for training (per device). \
    `--lr` is the learning rate used for training. \
    `--n_epochs` is the umber of training epochs. \
    `--max_steps` is the maximum number of training steps. \
    `--seed` is the random seed used for the specific run. \
    `--reparameterization` specifies whether to use reparameterization for prompt embeddings. \
    `--reparam_hidden_size` is the hidden size for reparameterization network. \
    `--model_freezing` specifies whether all model parameters should be frozen. \


To test the prompt, run the following:
```
python src/run_evaluation.py \
    --target_langs deu_Latn fra_Latn spa_Latn \
    --source_lang en \
    --prompt_length 5 \
    --model_name xglm-564M \
    --n_train_samples_per_class 8 \    
    --per_device_train_batch_size 8 \
    --lr 5e-3 \
    --seed 0 \
    --reparameterization \
    --reparam_hidden_size 200 \
    --model_freezing
```

The SIB-200 dataset ([Adelani et al., 2023](https://arxiv.org/abs/2309.07445)) used in our experiments is available [here](https://github.com/dadelani/sib-200).