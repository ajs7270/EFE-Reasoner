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
                ) -> tuple[torch.Tensor, torch.Tensor] :    # [[B, T, N_O + 1], [B, T, A, N_D + 1]]
                                                            # : Operator, Operand prediction, + 1 is padding

                self.number_vector = None  # [N_Q, H] N_Q : number of quantity (maybe use torch.gather?)

