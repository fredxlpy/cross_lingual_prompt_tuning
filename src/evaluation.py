from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sklearn
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# Import modules from local source if available, otherwise import from global scope
try:
    from src.preprocessing import preprocess_data
    from src.prompt import SoftPrompt
except ImportError:
    from preprocessing import preprocess_data
    from prompt import SoftPrompt


def evaluate_prompt(
    model,
    prompt,
    target_langs: list[str] = 'deu_Latn',
    model_name: str = 'xlm-roberta-base',
) -> tuple:

    # Clear GPU cache
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    if "xglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
    elif "bloom" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'bigscience/{model_name}')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})

    if model is None:
        if "xglm" in model_name:
            model = AutoModelForCausalLM.from_pretrained(f'facebook/{model_name}')
        elif "bloom" in model_name:
            model = AutoModelForCausalLM.from_pretrained(f'bigscience/{model_name}')

        # Untie static word embedding matrices of input and output layer
        model.config.tie_word_embeddings = False

        model.resize_token_embeddings(len(tokenizer))

        # Adapt input layer
        s_wte = SoftPrompt(
            model.get_input_embeddings(),
            prompt_length=prompt.shape[0],
            prompt_token_id=tokenizer.additional_special_tokens_ids[0],
            reparameterization=False,
            reparam_hidden_size=0)

        model.set_input_embeddings(s_wte)
        if "xglm" in model_name:
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.tensor(prompt))
        elif "bloom" in model_name:
            model.transformer.word_embeddings.weight = torch.nn.Parameter(torch.tensor(prompt))

    model = model.to(device)
    model.eval()

    accuracies, f1_scores = [], []

    for target_lang in target_langs:

        if "xglm" in model_name:
            label_to_token_dict = pd.read_excel('Data/sib-200_labels.xlsx', "xglm", index_col='original').to_dict()['en']
        elif "bloom" in model_name:
            label_to_token_dict = pd.read_excel('Data/sib-200_labels.xlsx', "bloom", index_col='original').to_dict()['en']

        # Load data
        test_set = Dataset.from_pandas(pd.read_table(f'Data/sib-200/{target_lang}/test.tsv'))
        test_set = test_set.remove_columns('index_id')
        test_set = test_set.map(lambda x: {'label': x['category']})
        test_set = test_set.map(lambda x: {'category': label_to_token_dict[x['category']]})
        test_set = test_set.class_encode_column('label')

        # Preprocess data
        test_set = test_set.map(lambda x: preprocess_data(
            x, tokenizer=tokenizer, prompt_length=prompt.shape[0], max_seq_length=128
        ), batched=True)

        test_set.set_format("torch")

        data_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=32 if torch.cuda.is_available() else 4,
                                                  shuffle=False)

        token_id_to_class_dict = {tokenizer(l, add_special_tokens=False)['input_ids'][0]:i for i,l in enumerate(label_to_token_dict.values())}

        labels = torch.tensor([], device=device)
        logits = torch.tensor([], device=device)
        loss = 0
        for batch in tqdm(data_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            mask_token_indices = torch.where(label != -100)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                logits = torch.concat([logits, out.logits[mask_token_indices]])
                labels = torch.concat([labels, label[mask_token_indices]])
                loss += torch.nn.CrossEntropyLoss()(out.logits[mask_token_indices], label[mask_token_indices])

        labels = labels.cpu().detach().numpy()
        labels = [token_id_to_class_dict[l] for l in labels]
        predictions = torch.argmax(logits[:, list(token_id_to_class_dict.keys())], dim=1).cpu().detach().numpy()

        acc = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)
        f1 = sklearn.metrics.f1_score(y_true=labels, y_pred=predictions, average='macro')

        accuracies.append(acc)
        f1_scores.append(f1)

        print('\n########################\nInference method 1\n########################')
        print(f'Target language: {target_lang}')
        print(f'Accuracy: {acc}')
        print(f'F1 score: {f1}')
        print(f'Loss: {loss/len(data_loader)}')

    return accuracies, f1_scores
