import numpy as np
import argparse
import torch

try:
    from src.training import train_prompt
except ImportError:
    from training import train_prompt

# Create an argument parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("--source_lang", type=str)  # e.g., 'en, 'de',...
parser.add_argument("--prompt_length", type=int, default=[10, 20, 50], nargs='+')
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_train_samples_per_class", type=int, default=[32, 512], nargs='+')
parser.add_argument("--per_device_train_batch_size", type=int, default=[8], nargs='+')
parser.add_argument("--lr", type=float, default=[5e-3], nargs='+')
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--seed", type=int, default=[0, 1, 2, 3], nargs='+')
parser.add_argument('--reparameterization', action='store_true')
parser.add_argument('--reparam_hidden_size', type=int, default=[0], nargs='+')
parser.add_argument("--model_freezing", action='store_true')

args = parser.parse_args()

SOURCE_LANG = args.source_lang
MODEL_NAME = args.model_name
N_EPOCHS = args.n_epochs
MAX_STEPS = args.max_steps
REPARAMETERIZATION = args.reparameterization
MODEL_FREEZING = args.model_freezing

results = []

for N_TRAIN_SAMPLES_PER_CLASS in args.n_train_samples_per_class:

    for LEARNING_RATE in args.lr:

        for PER_DEVICE_TRAIN_BATCH_SIZE in args.per_device_train_batch_size:

            for PROMPT_LENGTH in args.prompt_length:

                for REPARAM_HIDDEN_SIZE in args.reparam_hidden_size:

                    for SEED in args.seed:

                        torch.cuda.empty_cache()

                        trainer, prompt = train_prompt(
                            source_lang=SOURCE_LANG,
                            prompt_length=PROMPT_LENGTH,
                            model_name=MODEL_NAME,
                            n_train_samples_per_class=N_TRAIN_SAMPLES_PER_CLASS,
                            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                            learning_rate=LEARNING_RATE,
                            n_epochs=N_EPOCHS,
                            max_steps=MAX_STEPS,
                            seed=SEED,
                            reparameterization=REPARAMETERIZATION,
                            reparam_hidden_size=REPARAM_HIDDEN_SIZE,
                            model_freezing=MODEL_FREEZING)

                        # Save prompt
                        np.save(
                            f'./Prompts/prompt_MODEL_{MODEL_NAME}_SRC-{SOURCE_LANG}_NTRAIN-{N_TRAIN_SAMPLES_PER_CLASS}_PL-{PROMPT_LENGTH}_BS-{PER_DEVICE_TRAIN_BATCH_SIZE}-LR-{LEARNING_RATE}_RUN-{SEED}{f"_RP_{REPARAM_HIDDEN_SIZE}" if REPARAMETERIZATION else ""}{"_MF" if MODEL_FREEZING else ""}.npy',
                            prompt)

                        if not MODEL_FREEZING:
                            # Save model
                            torch.save(trainer.model, f'Models/prompt_MODEL_{MODEL_NAME}_SRC-{SOURCE_LANG}_NTRAIN-{N_TRAIN_SAMPLES_PER_CLASS}_PL-{PROMPT_LENGTH}_BS-{PER_DEVICE_TRAIN_BATCH_SIZE}-LR-{LEARNING_RATE}_RUN-{SEED}{f"_RP_{REPARAM_HIDDEN_SIZE}" if REPARAMETERIZATION else ""}')
