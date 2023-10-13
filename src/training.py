from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from datasets import DatasetDict, Dataset
import torch
import pandas as pd
import numpy as np

# Import modules from local source if available, otherwise import from global scope
try:
    from src.preprocessing import preprocess_data
    from src.trainers import MyTrainer
    from src.metrics import compute_metrics, preprocess_logits_for_metrics
    from src.prompt import SoftPrompt
except ImportError:
    from preprocessing import preprocess_data
    from metrics import compute_metrics, preprocess_logits_for_metrics
    from trainers import MyTrainer
    from prompt import SoftPrompt


def train_prompt(
    source_lang: str = 'en',
    prompt_length: int = 10,
    model_name: str = 'xglm-564M',
    n_train_samples_per_class: int or None = None,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 1e-5,
    n_epochs: int = 1,
    max_steps: int = 100,
    seed: int or None = None,
    reparameterization: bool = False,
    reparam_hidden_size: int = 50,
    model_freezing: bool = True
) -> tuple:
    """
    Trains a prompt for a given model.

    Parameters:
    - source_lang (str): Language code of the source data. Default is 'en'.
    - prompt_length (int): Length of the soft prompt. Default is 10.
    - model_name (str): Name of the model. Default is 'xglm-564M'.
    - n_train_samples_per_class (int or None): Number of training samples per class.
                                               If None, all samples are used. Default is None.
    - per_device_train_batch_size (int): Batch size for training. Default is 8.
    - learning_rate (float): Learning rate for training. Default is 1e-5.
    - n_epochs (int): Number of epochs for training. Default is 1.
    - max_steps (int): Maximum number of steps for training. Default is 100.
    - seed (int or None): Seed for reproducibility. If None, 42 is used. Default is None.
    - reparameterization (bool): Whether to use reparameterization. Default is False.
    - reparam_hidden_size (int): Hidden size for reparameterization network. Default is 50.
    - model_freezing (bool): Whether to freeze the model during training. Default is True.

    Returns:
    - trainer (MyTrainer): Trainer object after training.
    - prompt (numpy.ndarray): Trained prompt.
    """

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Load tokenizer and verbalizer
    if "xglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
        label_to_token_dict = pd.read_excel('Data/sib-200_labels.xlsx', "xglm", index_col='original').to_dict()['en']
    elif "bloom" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'bigscience/{model_name}')
        label_to_token_dict = pd.read_excel('Data/sib-200_labels.xlsx', "bloom", index_col='original').to_dict()['en']
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})

    # Undersampling
    train_set = pd.read_table(f'Data/sib-200/{source_lang}/train.tsv')
    train_set['label'] = train_set['category']
    min_samples = train_set['label'].value_counts().min()
    train_set = train_set.groupby('label').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

    val_set = pd.read_table(f'Data/sib-200/{source_lang}/dev.tsv')
    val_set['label'] = val_set['category']
    min_samples = val_set['label'].value_counts().min()
    val_set = val_set.groupby('label').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

    # Preprocess dataset
    dataset = DatasetDict({'train': Dataset.from_pandas(train_set),
                           'validation': Dataset.from_pandas(val_set)})
    dataset = dataset.remove_columns('index_id')
    dataset = dataset.map(lambda x: {'label': x['category']})
    dataset = dataset.map(lambda x: {'category': label_to_token_dict[x['category']]})
    dataset = dataset.class_encode_column('label')

    # Randomly select training and validation subsets
    if n_train_samples_per_class is not None:
        dataset['train'] = dataset['train'].train_test_split(
            train_size=len(label_to_token_dict) * n_train_samples_per_class,
            stratify_by_column='label', seed=None
        )['train']

        if n_train_samples_per_class!=32:
            n_val_samples_per_class = int(np.ceil(n_train_samples_per_class / 4))
            dataset['validation'] = dataset['validation'].train_test_split(
                train_size=len(label_to_token_dict) * n_val_samples_per_class,
                stratify_by_column='label', seed=None
            )['train']

    # Preprocess data
    dataset = dataset.map(lambda x: preprocess_data(
        x, tokenizer=tokenizer, prompt_length=prompt_length, max_seq_length=128
    ), batched=True)

    # Create model initialization function
    def model_init():
        # Load model
        if "xglm" in model_name:
            model = AutoModelForCausalLM.from_pretrained(f'facebook/{model_name}')
        elif "bloom" in model_name:
            model = AutoModelForCausalLM.from_pretrained(f'bigscience/{model_name}')

        # Untie static word embedding matrices of input and output layer
        model.config.tie_word_embeddings = False

        model.resize_token_embeddings(len(tokenizer))

        s_wte = SoftPrompt(
            model.get_input_embeddings(),
            prompt_length=prompt_length,
            prompt_token_id=tokenizer.additional_special_tokens_ids[0],
            reparameterization=reparameterization,
            reparam_hidden_size=reparam_hidden_size)

        model.set_input_embeddings(s_wte)

        model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings().wte)

        if model_freezing:
            # Freeze model (except prompt)
            parameters = list(model.parameters())
            for x in parameters[1:]:
                x.requires_grad = False

        # If prompt is reparameterized, the reparameterization network is unfrozen
        if reparameterization:
            parameters = list(model.parameters())
            for x in parameters[2:6]:
                print(x.shape)
                x.requires_grad = True

        return model

    # Define hyper-parameters
    training_args = TrainingArguments(output_dir=f"trainer/{source_lang}_{model_name}_{seed}",
                                      max_steps=max_steps,
                                      num_train_epochs=n_epochs,

                                      evaluation_strategy="epoch",

                                      save_strategy="epoch",
                                      save_total_limit=1,

                                      logging_strategy='steps',
                                      logging_steps=5,
                                      logging_first_step=True,
                                      logging_dir='logs',

                                      learning_rate=learning_rate,
                                      per_device_train_batch_size=per_device_train_batch_size,
                                      gradient_accumulation_steps=1,
                                      per_device_eval_batch_size=32 if torch.cuda.is_available() else 8,

                                      seed=seed if seed is not None else 42,
                                      warmup_steps=0,
                                      do_train=True,
                                      do_eval=True,
                                      report_to="none",
                                      optim='adamw_torch',
                                      warmup_ratio=0.1,

                                      load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss'
                                      )

    trainer = MyTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=lambda x: compute_metrics(x),
            preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(
                logits, labels)
        )

    # Train
    trainer.train()

    # Extract trained prompt
    if reparameterization:
        if "xglm" in model_name:
            prompt = trainer.model.model.embed_tokens.mlp(
                trainer.model.model.embed_tokens.weight
            ).detach().cpu().numpy()
        elif "bloom" in model_name:
            prompt = trainer.model.transformer.word_embeddings.mlp(
                trainer.model.transformer.word_embeddings.weight
            ).detach().cpu().numpy()
    else:
        prompt = list(trainer.model.parameters())[0].detach().cpu().numpy()

    return trainer, prompt
