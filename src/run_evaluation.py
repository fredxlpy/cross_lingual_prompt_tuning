import numpy as np
import argparse
import pandas as pd
import torch

try:
    from src.evaluation import evaluate_prompt
except ImportError:
    from evaluation import evaluate_prompt


# Create an argument parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("--target_langs", type=str, default=['eng_Latn'], nargs='+')
parser.add_argument("--source_lang", type=str)  # e.g., 'en, 'de',...
parser.add_argument("--prompt_length", type=int, default=[10, 20, 50], nargs='+')
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_train_samples_per_class", type=int, default=[32, 512], nargs='+')
parser.add_argument("--per_device_train_batch_size", type=int, default=[8], nargs='+')
parser.add_argument("--lr", type=float, default=[5e-3], nargs='+')
parser.add_argument("--seed", type=int, default=[0, 1, 2, 3], nargs='+')
parser.add_argument('--reparameterization', action='store_true')
parser.add_argument('--reparam_hidden_size', type=int, default=[0], nargs='+')
parser.add_argument("--model_freezing", action='store_true')

args = parser.parse_args()

TARGET_LANGS = args.target_langs
SOURCE_LANG = args.source_lang
MODEL_NAME = args.model_name
REPARAMETERIZATION = args.reparameterization
MODEL_FREEZING = args.model_freezing


all_results = pd.DataFrame()

for N_TRAIN_SAMPLES_PER_CLASS in args.n_train_samples_per_class:

    for LEARNING_RATE in args.lr:

        for PER_DEVICE_TRAIN_BATCH_SIZE in args.per_device_train_batch_size:

            for PROMPT_LENGTH in args.prompt_length:

                for REPARAM_HIDDEN_SIZE in args.reparam_hidden_size:

                    for SEED in args.seed:

                        torch.cuda.empty_cache()

                        prompt = np.load(
                            f'./Prompts/prompt_MODEL_{MODEL_NAME}_SRC-{SOURCE_LANG}_NTRAIN-{N_TRAIN_SAMPLES_PER_CLASS}_PL-{PROMPT_LENGTH}_BS-{PER_DEVICE_TRAIN_BATCH_SIZE}-LR-{LEARNING_RATE}_RUN-{SEED}{f"_RP_{REPARAM_HIDDEN_SIZE}" if REPARAMETERIZATION else ""}{"_MF" if MODEL_FREEZING else ""}.npy')

                        if not MODEL_FREEZING:
                            model = torch.load(f'Models/prompt_MODEL_{MODEL_NAME}_SRC-{SOURCE_LANG}_NTRAIN-{N_TRAIN_SAMPLES_PER_CLASS}_PL-{PROMPT_LENGTH}_BS-{PER_DEVICE_TRAIN_BATCH_SIZE}-LR-{LEARNING_RATE}_RUN-{SEED}{f"_RP_{REPARAM_HIDDEN_SIZE}" if REPARAMETERIZATION else ""}')
                        else:
                            model = None

                        eval_result = evaluate_prompt(
                            model,
                            prompt,
                            target_langs=TARGET_LANGS,
                            model_name=MODEL_NAME)

                        results = pd.DataFrame()
                        results['accuracy'] = eval_result[0]
                        results['f1'] = eval_result[1]
                        results['source_lang'] = SOURCE_LANG
                        results['target_lang'] = TARGET_LANGS
                        results['model'] = MODEL_NAME
                        results['prompt_length'] = PROMPT_LENGTH
                        results['batch_size'] = PER_DEVICE_TRAIN_BATCH_SIZE
                        results['lr'] = LEARNING_RATE
                        results['n_train'] = N_TRAIN_SAMPLES_PER_CLASS
                        results['model_freezing'] = MODEL_FREEZING
                        results['reparameterization'] = REPARAMETERIZATION
                        if REPARAMETERIZATION:
                            results['reparam_hidden_size'] = REPARAM_HIDDEN_SIZE
                        results['seed'] = SEED

                        all_results = pd.concat([all_results, results])

                        all_results.to_excel(f'./Output/results_{MODEL_NAME}_{SEED}.xlsx', index=False)
