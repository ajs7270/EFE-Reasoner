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

                self.number_vector = self._get_num_vec(input, number_mask)  # [N_Q, H] N_Q : number of quantity (maybe use torch.gather?)

    def _get_num_vec(self, input: torch.Tensor, number_mask: torch.Tensor, concat: bool) -> torch.Tensor:
        # input : [B, S, H]
        # number_mask : [B, S]
        # number_vector : [N_Q, H] or [N_Q, H*2] regarding concat
        # returns number vector for each quantity
        n = int(torch.max(number_mask).item())
        if concat:
            number_vector = torch.zeros(n, self.hidden_dim*2)
        else:
            number_vector = torch.zeros(n, self.hidden_dim)
        for i in range(n):
            index = torch.nonzero(number_mask == i)
            first = input[index[0, 0], index[0, 1], :]   # [H]
            last = input[index[-1, 0], index[-1, 1], :]  # [H]
            assert first.shape == last.shape == (self.hidden_dim,)
            if concat:
                number_vector[i, :] = torch.cat((first, last), dim=0)  # [H*2]
            else:
                number_vector[i, :] = first + last

        return number_vector
