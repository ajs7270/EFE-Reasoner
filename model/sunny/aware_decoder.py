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
        x = x + self.pe[:, :x.size(1), :]
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
        self.operand_transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # fc layer that projects the operator + operand vectors into a length H vector
        self.equation_result_projection = nn.Linear(self.hidden_dim * (self.max_arity + 1), self.hidden_dim)

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
    def _get_operand_vector(self, batch_idx:int , idx: int)-> torch.Tensor:
        """
        get operand vector in a problem (not batch)
        :param idx: index of operand
        :param number_vector: number vector (LM caching) in a problem [N_Q, H]
        :return: operand vector [H]
        """
        # fetch constant vector
        if 0 <= idx < self.const_num:
            return self.operand_projection(self.const_vector[idx, :])  # [H]

        # fetch number vector
        idx -= self.const_num
        if 0 <= idx < self.max_number_size:
            return self.operand_projection(self.number_vector[batch_idx, idx, :]) # [H]

        # fetch previous result vector
        idx -= self.max_number_size
        if 0 <= idx < self.max_equation:
            return self.previous_result_vector[batch_idx, idx, :]

    def get_operand_vector(self, i, j, operand_idx: torch.Tensor) -> torch.Tensor:
        # operand_index : [B, T, A]
        # operand_vector : [B, T, A, H]
        operand_idx = operand_idx[:,i,j].clone()
        batch_size, = operand_idx.size()
        operands_prediction_vectors = torch.zeros(batch_size, self.hidden_dim, device=operand_idx.device)
        for batch_idx in range(batch_size):
            operands_prediction_vectors[batch_idx, :] = self._get_operand_vector(operand_idx[batch_idx])
        return operands_prediction_vectors

    def get_arity_vector(self, index: torch.Tensor) -> torch.Tensor:
        """
        get arity vector in a problems(parallel)

        :param index: [B, A]
        :return: [B, A, H]
        """
        batch_size, _ = index.size()
        arity_vector = torch.zeros(batch_size, self.max_arity, self.hidden_dim, device=index.device)
        for batch_idx in range(batch_size):
            for i in range(self.max_arity):
                arity_vector[batch_idx, i, :] = self._get_operand_vector(batch_idx, index[batch_idx, i].item())
        return arity_vector

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
        self.number_vector = torch.zeros(input.size(0), self.max_number_size,
                                         self.hidden_dim * 2 if self.concat else self.hidden_dim)
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
        batch_sos_embedding = self.embedding(torch.LongTensor([0]).to(device)).unsqueeze(dim=0).repeat(input.size(0), 1, 1)
        assert batch_sos_embedding.size() == (input.size(0), 1, self.hidden_dim)

        # gold_operator_vectors : [B, T, H]
        gold_operator_vectors = torch.cat([batch_sos_embedding, gold_operator_vectors], dim=1)  # [B, T+1[sos], H]
        # add position encoding
        gold_operator_vectors = self.pos_encoder(gold_operator_vectors)  # [B, T+1[sos], H]
        # tgt_mask(attention mask) : [T, T] => [T+1[sos], T+1[sos]]
        # attention mask는 batch size를 신경쓰지 않고 넣어주고, batch size만큼 reshape 해주는 과정은 transformer forward 함수 내부에서 진행
        tgt_mask = self.generate_square_subsequent_mask(self.max_equation + 1).to(device)  # [T+1[sos], T+1[sos]]

        operator_output = self.operator_transformer(
            tgt= gold_operator_vectors, # [B, T+1[sos], H]
            memory=input, # [B, S, H]
            tgt_mask=tgt_mask, # [T+1[sos], T+1[sos]]
            tgt_key_padding_mask=tgt_key_padding_mask, # [B, T+1[sos]]
            memory_key_padding_mask=memory_key_padding_mask # [B, S]
        ) # -> [B, T+1[eos], H]

        operators_logit = self.operator_classifier(operator_output)  # [B, T+1[eos], H] -> [B, T+1[eos], N_O]

        # 2. Operand prediction
        # Operator가 다 계산돼서 이제 각 operator마다 operand를 계산해야 하는데,
        # 그렇게 하기 위해서 각 배치마다 결과를 모아서 tgt로 넣어주어야 한다.
        # 두 가지 방식이 존재할 수 있다.
        # 1. (데이터셋 하나씩 끝내는 방식)하나의 배치에 있는 모든 operator에 대해 한번에 값을 계산할 수 있는 방식
        # 2. (병렬로 끝내는 방식) => (우리의 선택)
        # operand는 전체를 다 계산하고, 불필요한 부분은 loss 계산에서 제외한다.

        # Operand Transformer Input Shape
        # memory : [B, S, H] Question context
        # input : [B, A+1[OP-start], H] Output of operator prediction ([SOS] token is not included)
        # output : [B, A+1[OP-start], N_D]
        # Operand Transformer를 T번 통과시킴 -> T번 통과시키면서 얻은 값들은 concat해서 리턴해줌

        # eg. operand
        # tgt = ["+", "2", "3", "", "-", "3", "4", "", "", "", "", ""]
        # 중요한 건 <sos>에 positional encoding을 더해주는 것 => 그래야지 operand transformer가 몇 번째 operator를 평가하는지 알 수 있음
        # 그래서 굳이 앞에 연산자와 피연산자들을 더해줄 필요는 없다.

        for i in range(self.max_equation):
            gold_operand_vectors = torch.zeros(input.size(0), self.max_arity + 1, self.hidden_dim).to(device)
            # 각 배치의 i번째 operator에 대한 gold operand vector를 가져옴
            operand_sos_vector = self.operator_projection(torch.index_select(self.operator_vector, dim=0,
                                                              index=gold_operators[:, i])).unsqueeze(dim=1)
            operand_arity_vectors = self.get_arity_vector(index=gold_operands[:, i, :])
            gold_operand_vectors = torch.cat([operand_sos_vector, operand_arity_vectors] , dim=1)
            gold_operand_vectors = self.pos_encoder(gold_operand_vectors)  # [B, A+1[sos], H]

            tgt_mask = self.generate_square_subsequent_mask(self.max_arity + 1).to(device)  # [B, T*A+1[sos], T*A+1[sos]]
            # tgt_key_padding_mask : [B, T] => [B, T+1[sos]]
            tgt_key_padding_mask = (gold_operands[:,i,:]== none_idx)  # [B, T]  [False, False False, ..., True, True, True]
            tgt_key_padding_mask = torch.cat([torch.zeros(input.size(0), 1, device=device).bool(), tgt_key_padding_mask], dim=1)

            operand_output = self.operand_transformer(
                tgt = gold_operand_vectors, # [B, A+1[sos], H]
                memory= input, # [B, S, H]
                tgt_mask= tgt_mask, # [B, A+1[sos], A+1[sos]] tgt_mask가 제대로 동작할까? => 우리는 일반적인 transformer 동작 방식이 아님
                tgt_key_padding_mask= tgt_key_padding_mask,# [B, A+1[sos]]
                memory_key_padding_mask=memory_key_padding_mask # [B, S]
            ) # => [B, A+1[eos], H]

            # operand_output을 이용해서 operand_logit을 만들어야 한다.
            for j in range(self.max_arity):
                operand_logit = self.operand_classifier(
                    operand_output[:, j, :])  # [B, 1(Sequence Length), H] -> [B, N_D + 1]
                operands_logit[:, i, j, :] = operand_logit  # loss를 구할때는 logit에 softmax 값을 취한 것을 CE Loss로 구해야함

            # reshape operand output vector of shape [B, A+1[eos], H] => [B, (A + 1) * H]
            operand_output = operand_output.reshape(operand_output.size(0), -1)
            # update previous_result_vector [B, T, H]
            self.previous_result_vector[:, i, :] = self.equation_result_projection(operand_output) # [B, H]

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
