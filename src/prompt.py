import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 prompt_length: int = 10,
                 random_range: float = 0.5,
                 prompt_token_id: int = 1,
                 reparameterization: bool = False,
                 reparam_hidden_size: int = 50):

        """
        SoftPrompt module for incorporating soft prompt embeddings into text encoding.

        Args:
            wte (nn.Embedding): Static word token embeddings.
            prompt_length (int): Number of task-specific embedding tokens.
            random_range (float): Range for initializing the task-specific embedding weights.
            prompt_token_id (int): Token ID for identifying prompt positions.
            reparameterization (bool): Whether to use reparameterization for learned embeddings.
            reparam_hidden_size (int): Size of the hidden layer in the reparameterization network.
        """

        super(SoftPrompt, self).__init__()
        self.wte = wte
        self.n_tokens = prompt_length
        self.prompt_token_id = prompt_token_id
        self.weight = nn.parameter.Parameter(
            torch.FloatTensor(prompt_length, wte.weight.size(1)).uniform_(-random_range, random_range))
        if reparameterization:
            self.mlp = self._build_mlp(wte.weight.size(1), reparam_hidden_size)
        self.reparameterization = reparameterization

    def _build_mlp(self, input_size, hidden_size):
        """
        Build the reparameterization MLP.

        Args:
            input_size (int): Input size of the MLP.
            hidden_size (int): Hidden size of the MLP.

        Returns:
            MLP: Reparameterization MLP.
        """
        mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size)
        )
        return mlp

    def forward(self, tokens):
        """
        Forward pass of the SoftPrompt module.

        Args:
            tokens (torch.LongTensor): Input tokens before encoding.

        Returns:
            torch.FloatTensor: Encoding of text concatenated with learned task-specific embedding.
        """
        input_embedding = self.wte(tokens)

        if self.reparameterization:
            learned_embedding = self.mlp(self.weight)
        else:
            learned_embedding = self.weight

        for i in range(tokens.size(0)):
            prompt_indices = torch.where(tokens[i] == self.prompt_token_id)[0]
            input_embedding[i, prompt_indices] = learned_embedding

        return input_embedding