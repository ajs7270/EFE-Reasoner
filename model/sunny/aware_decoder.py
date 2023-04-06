import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AwareDecoder(nn.Module):
    def __init__(self,
                 input_hidden_dim: int,
                 num_layers: int,
                 operator_vector: torch.Tensor, # PAD(None) + OPERATOR
                 const_vector: torch.Tensor,    # CONST
                 operator_num: int,     # OPERATOR : PAD(None) + OPERATOR
                 const_num: int,        # OPERAND : PAD(None) + CONST
                 max_number_size: int,  # OPERAND : NUMBER (max count in the question)
                 max_equation: int,     # OPERAND : PREVIOUS_RESULT (max count in equations)
                 max_arity: int,
                 label_pad_id: int,     # (OperatorLabelEncoder and OperandLabelEncoder)'s pad id
                 tokenizer_pad_id: int, # (Tokenizer)'s pad id
                 concat: bool = True):
        super().__init__()
        # configuration setting
        self.hidden_dim = input_hidden_dim
        self.num_layers = num_layers
        self.operator_num = operator_num
        self.label_pad_id = label_pad_id
        self.tokenizer_pad_id = tokenizer_pad_id
        self.const_num = const_num
        self.max_equation = max_equation
        self.max_number_size = max_number_size
        self.max_arity = max_arity
        self.concat = concat

        # operand candidate vector : const vector
        # N_C : number of constant
        self.const_vector = nn.Parameter(const_vector)  # [N_C, H] or [N_C, H*2] regarding concat
        assert self.const_num == self.const_vector.size(0)

        # operator candidate vector
        # N_O : number of operator
        self.operator_vector = nn.Parameter(operator_vector)    # [N_O, H] or [N_O, H*2] regarding concat
        assert self.operator_num == self.operator_vector.size(0)
        self.embedding = nn.Embedding(1, self.hidden_dim)       # for [SOS] token embedding

        # positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout=0.1)

        # operator, operand projection layer
        hidden_dim = self.hidden_dim * 2 if self.concat else self.hidden_dim
        self.operator_projection = nn.Sequential(
            nn.Linear(hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.operand_projection = nn.Sequential(
            nn.Linear(hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # operator classifier
        self.operator_classifier = nn.Sequential(
          nn.Linear(self.hidden_dim, self.operator_num)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            batch_first=True,
        )
        self.operator_transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # operand classifier : operand는 여러개가 뽑혀야 하므로 일반적인 classifier를 사용해서는 안됨 => gru같은 neural network을 사용해야 함
        self.operand_gru = nn.GRU(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,  # Layer를 쌓으면 학습이 잘 되지 않으므로 Multi Layer GRU는 사용하지 않는다.
                                  bidirectional=False,  # 우리는 과거와 현재를 굳이 확인할 필요가 없으므로
                                  batch_first=True)

        # 어떤 정보를 계산해야 하는지를 남겨준다 (hidden state에 계산한 결과를 계속 넣어줘서, 그 정보는 없애도록 학습되길 기대한다)
        self.context_gru = nn.GRU(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers,
                                  bidirectional=False,
                                  batch_first=True)

        self.operand_classifier = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                (self.const_num + self.max_number_size + self.max_equation) // 2
            ),
            nn.ReLU(),
            nn.Linear(
                (self.const_num + self.max_number_size + self.max_equation) // 2,
                self.const_num + self.max_number_size + self.max_equation # PAD(None) + CONST + NUM + PREVIOUS RESULT
            )
        )

        # context attention
        # self.context_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        # self.question_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)

        # dummy classifier
        # deductive reasoner와 같은 방식을 사용하지 않기 때문에 dumy classifier가 소용이 없다.
        # self.dummy_classifier = nn.Linear(self.hidden_dim, 2) # dummy or not을 classification

    def get_operand_vector(self, i, j, operand_idx: torch.Tensor) -> torch.Tensor:
        # operand_index : [B, T, A]
        # operand_vector : [B, T, A, H]
        operand_idx = operand_idx[:,i,j].clone()
        batch_size, = operand_idx.size()
        operands_prediction_vectors = torch.zeros(batch_size, self.hidden_dim, device=operand_idx.device)
        for batch_idx in range(batch_size):
            if 0 <= operand_idx[batch_idx] < self.const_num:
                operands_prediction_vectors[batch_idx, :] = self.operand_projection(
                    self.const_vector[operand_idx[batch_idx], :])  # [H]

            # fetch number vector
            operand_idx[batch_idx] -= self.const_num
            if 0 <= operand_idx[batch_idx] < self.max_number_size:
                operands_prediction_vectors[batch_idx, :] = self.operand_projection(
                    self.number_vector[batch_idx, operand_idx[batch_idx], :])

            # fetch previous result vector
            operand_idx[batch_idx] -= self.max_number_size
            if 0 <= operand_idx[batch_idx] < self.max_equation:
                operands_prediction_vectors[batch_idx, :] = self.previous_result_vector[batch_idx,
                                                                  operand_idx[batch_idx], :]
        return operands_prediction_vectors

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                # B : Batch size
                # S : Max length of tokenized problem text (Source)
                # H : Hidden dimension of language model
                # A : Max number of arity of operator
                # N_O : Number of operators (with PAD(None))
                # N_D : Number of operands (with PAD(None))
                # N_Q : number of quantity
                # T : Max length of tokenized equation (Target)
                input: torch.Tensor,  # [B, S, H]: Language model output
                attention_mask: torch.Tensor,  # [B, S] : Language model attention mask
                question_mask: torch.Tensor,  # [B, S] : Language model question mask
                number_mask: torch.Tensor,  # [B, S] : Language model number mask
                gold_operators: torch.Tensor,  # [B, T+1] : Gold operator
                gold_operands: torch.Tensor,  # [B, T+1, A] : Gold operand
                ) -> tuple[torch.Tensor, torch.Tensor]:  # [[B, T, N_O], [B, T, A, N_D]] : Operator, Operand logit
        # : Operator, Operand prediction, + 1 is padding
        # operand candidate vector setup : const vector(dataset 만들때 생성해서, 이 모델의 init 단계에서 설정), number vector, previous result vector
        # self.number_vector = torch.zeros(input.size(0), self.max_number_size,
        #                                  self.hidden_dim * 2 if self.concat else self.hidden_dim)
        device = input.device
        self.number_vector = self._get_num_vec(input, number_mask, self.max_number_size, concat=self.concat).to(device)  # [N_Q, H] or [N_Q, H*2]
        self.previous_result_vector = torch.zeros(input.size(0), self.max_equation, self.hidden_dim).to(device)  # [B, T, H]

        # Initialize return values
        operands_logit = torch.zeros(input.size(0), self.max_equation, self.max_arity,
                                     self.const_num + self.max_number_size + self.max_equation).to(device)  # PAD(None) + CONST + NUM + PREVIOUS RESULT
        gold_operator_vectors = torch.zeros(input.size(0), self.max_equation, self.hidden_dim).to(device)
        operands_prediction_vectors = torch.zeros(input.size(0), self.max_equation, self.max_arity, self.hidden_dim).to(device)


        # 1. Operator prediction
        # memory : input [B, S, H]
        # gold_operator_vectors : LM caching을 사용한 tgt Embedding [B, T, H]
        # operator_output : [B, T, H]
        # operator_logit : [B, T, N_O]
        none_idx = 0
        context_vector = input[:, 0, :]  # [B, H]

        for i in range(self.max_equation):
            gold_operator_vectors[:, i, :] = self.operator_projection(torch.index_select(self.operator_vector, dim=0,
                                                                       index=gold_operators[:, i]))  # [B, H]
        # tgt_key_padding_mask : [B, T] => [B, T+1[sos]]
        tgt_key_padding_mask = (gold_operators == none_idx)  # [B, T]  [False, False False, ..., True, True, True]
        tgt_key_padding_mask = torch.cat([torch.zeros(input.size(0), 1, device=device).bool(), tgt_key_padding_mask], dim=1)

        memory_key_padding_mask = (attention_mask != self.tokenizer_pad_id)  # [B, S] [False, False False, ..., True, True, True]

        # 1. <sos> token을 추가
        # <sos> token [1, H] => [B, 1, H]
        batch_sos_embedding = self.embedding(torch.Tensor([0]).to(device).long()).unsqueeze(dim=0).repeat(input.size(0), 1, 1)
        assert batch_sos_embedding.size() == (input.size(0), 1, self.hidden_dim)

        # gold_operator_vectors : [B, T, H]
        gold_operator_vectors = torch.cat([batch_sos_embedding, gold_operator_vectors], dim=1)  # [B, T+1[sos], H]
        # add position encoding
        gold_operator_vectors = self.pos_encoder(gold_operator_vectors)  # [B, T+1[sos], H]
        # tgt_mask(attention mask) : [T, T] => [T+1[sos], T+1[sos]]
        # attention mask는 batch size를 신경쓰지 않고 넣어주고, batch size만큼 reshape 해주는 과정은 transformer forward 함수 내부에서 진행
        tgt_mask = self.generate_square_subsequent_mask(self.max_equation + 1).to(device)  # [T+1[sos], T+1[sos]]

        operator_output = self.operator_transformer(
            tgt= gold_operator_vectors, # [B, T, H] => [B, T+1[sos], H]
            memory=input, # [B, S, H]
            tgt_mask=tgt_mask, # [B, T, T] => [B, T+1[sos], T+1[sos]]
            tgt_key_padding_mask=tgt_key_padding_mask, # [B, T+1[sos]]
            memory_key_padding_mask=memory_key_padding_mask # [B, S]
        )

        operators_logit = self.operator_classifier(operator_output)  # [B, T+1[eos], N_O]

        # 2. Operand prediction
        # operand는 전체를 다 계산하고, 불필요한 부분은 loss 계산에서 제외한다.
        for i in range(self.max_equation):
            for j in range(self.max_arity):
                if j == 0:
                    if self.training: # teacher forcing
                        x = torch.unsqueeze(
                            self.operator_projection(
                                torch.index_select(self.operator_vector, dim=0, index=gold_operators[:, i])),
                            dim=1
                        )
                    else:
                        x = torch.unsqueeze(operator_output[:, i, :], dim=1)  # [B, 1(Sequence Length), H]

                    hx = torch.unsqueeze(context_vector, dim=0).expand(self.num_layers, -1, -1)  # [num_layers , B, H]
                else:
                    if self.training: # teacher forcing
                        # operand의 경우 3가지 경우의 수에서 답을 가져와야 하기 때문에 함수로 빼야 한다.
                        x = torch.unsqueeze(
                            self.get_operand_vector(i,j - 1, gold_operands),
                            dim=1
                        )
                    else:
                        x = torch.unsqueeze(operands_prediction_vectors[:, i, j - 1, :], dim=1)  # [B, 1(Sequence Length), H]

                    hx = hx  # previous hidden state

                x, hx = self.operand_gru(x, hx=hx.contiguous())  # [B, 1(Sequence Length), H], [1, B, H]

                # get operand vector
                operand_logit = self.operand_classifier(x.squeeze(dim=1))  # [B, N_D + 1]
                operands_logit[:, i, j, :] = operand_logit  # loss를 구할때는 logit에 softmax 값을 취한 것을 CE Loss로 구해야함
                operand_idx = torch.argmax(operand_logit, dim=1)

                for batch_idx in range(operand_logit.size(0)):
                    # fetch constant vector
                    if 0 <= operand_idx[batch_idx] < self.const_num:
                        # if current operand is None Vector
                        if operand_idx[batch_idx] == self.label_pad_id:
                            # 3. Update operand vector (#0, #1, #2등 이전 연산의 결과)
                            # 이전 연산결과를 저장해야 하는데, 처음 none이 등장했을 때만 저장
                            if max(self.previous_result_vector[batch_idx, i, :]).item() == 0:
                                self.previous_result_vector[batch_idx, i, :] = x[batch_idx, 0, :]

                        operands_prediction_vectors[batch_idx, i, j, :] = self.operand_projection(self.const_vector[operand_idx[batch_idx],:])  # [H]

                    # fetch number vector
                    operand_idx[batch_idx] -= self.const_num
                    if 0 <= operand_idx[batch_idx] < self.max_number_size:
                        operands_prediction_vectors[batch_idx, i, j, :] = self.operand_projection(self.number_vector[batch_idx, operand_idx[batch_idx],:])

                    # fetch previous result vector
                    operand_idx[batch_idx] -= self.max_number_size
                    if 0 <= operand_idx[batch_idx] < self.max_equation:
                        operands_prediction_vectors[batch_idx, i, j, :] = self.previous_result_vector[batch_idx,operand_idx[batch_idx], :]

            # 중간에 none operand가 등장하지 않는 경우 이번 연산 결과를 update
            for batch_idx in range(input.size(0)):  # batch size
                if max(self.previous_result_vector[batch_idx, i, :]).item() == 0:
                    # 마지막 operand의 예측값으로 update
                    self.previous_result_vector[batch_idx, i, :] = operands_prediction_vectors[batch_idx, i, -1, :]

        return operators_logit, operands_logit


    def _get_num_vec(self, input: torch.Tensor, number_mask: torch.Tensor, max_number: int, concat: bool) -> torch.Tensor:
        # input : [B, S, H]
        # number_mask : [B, S]
        # number_vector : [B, N_Q, H] or [B, N_Q, H*2] regarding concat
        # returns number vector for each quantity
        batch_size = input.size(0)

        if concat:
            number_vector = torch.zeros(batch_size, max_number, self.hidden_dim * 2)
        else:
            number_vector = torch.zeros(batch_size, max_number, self.hidden_dim)

        for i in range(batch_size):
            for j in range(max_number):
                index = torch.nonzero(number_mask[i] == j+1)

                # 발견된 위치가 없는 경우
                if index.size(0) == 0:
                    continue

                first = input[i, index[0, 0], :]  # [H]
                last = input[i, index[-1, 0], :]  # [H]
                # assert first.shape == last.shape == (self.hidden_dim,)
                if concat:
                    number_vector[i, j, :] = torch.cat([first, last], dim=0)  # [H*2]
                else:
                    number_vector[i, j, :] = torch.mean(first, last)  # [H]

        return number_vector
