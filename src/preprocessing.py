import datasets
import torch
from transformers import PreTrainedTokenizer


def preprocess_data(examples: dict,
                    tokenizer: PreTrainedTokenizer,
                    prompt_length: int = 2,
                    max_seq_length: int = 128
                    ) -> datasets.Dataset:
    """
    Preprocess and transforms NLI data into a masked language modeling task.

    Args:
        examples (dict): A dictionary containing the premise, hypothesis, and label.
        tokenizer (PreTrainedTokenizer): A tokenizer object used to tokenize the text pairs.
        prompt_length (int, optional): The length of the prompt placeholders. Default is 2.
        max_seq_length (int, optional): The maximum sequence length for padding/truncation. Default is 128.

    Returns:
        datasets.Dataset: Tokenized dataset with input_ids, attention_mask, and labels.

    """

    #####################################
    #         PRE-PROCESS INPUT         #
    #####################################

    # Tokenize the text pairs (premise & hypothesis) using the tokenizer
    tokenized_input = tokenizer(
        examples["text"],
        [tokenizer.additional_special_tokens[0] * prompt_length + tokenizer.eos_token] * len(examples["text"]),
        padding="max_length",
        max_length=max_seq_length, truncation='only_first', return_tensors='pt', add_special_tokens=False
    )

    ######################################
    #         PRE-PROCESS OUTPUT         #
    ######################################

    # Initialize lists for output
    input_ids, attention_mask, labels = [], [], []

     # Loop over the batch dimension
    for i in range(tokenized_input["input_ids"].size(0)):
        eos_token_index = tokenized_input['input_ids'][i] == tokenizer.eos_token_id
        eos_token_index = torch.concat([eos_token_index[1:], torch.Tensor([False])])
        eos_token_index = eos_token_index.type('torch.BoolTensor')
        # Prepare labels
        labels.append(
            torch.where(
                eos_token_index,
                tokenizer(examples['category'][i], add_special_tokens=False)['input_ids'][-1],
                -100
            ).detach().cpu().tolist()
        )

    tokenized_input['labels'] = torch.Tensor(labels).type(torch.int64)

    # Return tokenized dataset
    return tokenized_input
