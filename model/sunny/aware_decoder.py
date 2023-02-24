import torch
from torch import nn


class AwareDecoder(nn.Module):
    def __init__(self, input_hidden_dim: int):
        super().__init__()
        self.hidden_dim = input_hidden_dim

    def forward(self,
                # B : Batch size
                # S : Max length of tokenized problem text (Source)
                # H : Hidden dimension of language model
                # A : Max number of arity of operator
                # N_O : Number of operators
                # N_D : Number of operands
                # T : Max length of tokenized equation (Target)
                input: torch.Tensor,                        # [B, S, H]: Language model output
                attention_mask: torch.Tensor,               # [B, S] : Language model attention mask
                question_mask: torch.Tensor,                # [B, S] : Language model question mask
                number_mask: torch.Tensor,                  # [B, S] : Language model number mask
                ) -> tuple[torch.Tensor, torch.Tensor] :    # [[B, T, 1, N_O], [B, T, A, N_D]] : Operator, Operand prediction
        pass

