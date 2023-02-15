from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
import torch.utils.data as data
import json
from transformers import AutoTokenizer, AutoConfig
import torch
import re

BASE_PATH = Path(__file__).parent.parent


@dataclass
class Feature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    question_mask: torch.Tensor
    number_mask: torch.Tensor
    label: torch.tensor



@dataclass
class Problem:
    context: str
    question: str
    numbers: list[str]
    same_number_idx: list[list[int]]
    equation: list[list[str]]
    golden_op: list[str]
    golden_argument: list[list[str]]


class Dataset(data.Dataset):
    def __init__(self,
                 data_path: str = "data/processed/mathqa/train.json",
                 constant_path: str = "data/processed/mathqa/train_constant_list.json",
                 pretrained_model_name: str = "roberta-base",
                 ):
        with open(Path(BASE_PATH, data_path), 'r') as f:
            self.orig_dataset = json.load(f)
        with open(Path(BASE_PATH, constant_path), 'r') as f:
            constant_list = json.load(f)
        self.constant2id = {constant: i for i, constant in enumerate(constant_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.plm_config = AutoConfig.from_pretrained(pretrained_model_name)

        # 문제에 등장하는 모든 숫자를 <quant>로 치환한 후 tokenize하기 때문에 tokenize가 끝난 후 숫자의 위치를 찾기 위해 사용
        # this idea is from Deductive MWP
        self.quant_list_ids = self.tokenizer(" <quant> ", return_tensors="pt").input_ids[0][1:-2]

        self.features = []
        for problem_dict in tqdm(self.orig_dataset, desc="Converting Problem to Features "):
            problem = Problem(**problem_dict)
            feature = self._convert_to_feature(problem)
            self.features.append(feature)

        print(self.tokenizer)

    def _convert_to_feature(self, problem: Problem) -> Feature:
        # ~~~~ number0 ~~~~ number2 ~~~~~~~ 문장을
        # ~~~~ <quant> ~~~~ <quant> ~~~~~~~ 로 치환
        problem_text = problem.context + " " + problem.question
        problem_text = self._num2quent(problem_text)

        # tokenize
        tokenized_problem = self.tokenizer(problem_text, return_tensors="pt").input_ids
        tokenized_context = self.tokenizer(problem.context, return_tensors="pt").input_ids
        # 첫번째는 SOS, 마지막은 EOS 토큰이므로 제외시킴
        number_tensors = [self.tokenizer(number, return_tensors="pt").input_ids[:, 1:-1] for number in problem.numbers]

        tokenized_problem, question_mask, number_mask, num_count = self._translate2number(tokenized_problem,
                                                                                          tokenized_context,
                                                                                          number_tensors,
                                                                                          self.quant_list_ids)
        assert num_count == len(number_tensors), "number의 개수가 맞지 않음 {} != {}\n" \
                                                 "number list : {}\n" \
                                                 "tokenized result : {}" \
            .format(num_count, len(number_tensors), number_tensors,
                    self.tokenizer.convert_ids_to_tokens(tokenized_problem[0]))

    @classmethod
    def _translate2number(self, tokenized_problem, tokenized_context, number_tensors, quant_list_ids=None):
        question_mask = torch.zeros_like(tokenized_problem)  # shape (1,38)
        question_mask[:, tokenized_context.shape[1] - 1:] = 1
        number_mask = torch.zeros_like(tokenized_problem)  # shape (1,38)

        # <quant>를 number_tensors로 치환
        num_count = 0
        cur = 0
        while cur < len(tokenized_problem[0]) - len(quant_list_ids) + 1:
            if torch.equal(tokenized_problem[0][cur:cur + len(quant_list_ids)], quant_list_ids):
                # number_mask에 숫자의 등장순서에 따라 1,2,3으로 마스킹
                number_mask = torch.cat([number_mask[:, :cur],
                                         torch.full(number_tensors[num_count].shape, num_count + 1),
                                         number_mask[:, cur + len(quant_list_ids):]], dim=1)
                # question_mask 사이즈 조정
                question_mask = torch.cat([question_mask[:, :cur],
                                           torch.full(number_tensors[num_count].shape, question_mask[0, cur]),
                                           question_mask[:, cur + len(quant_list_ids):]], dim=1)
                # number_tensors로 치환
                tokenized_problem = torch.cat([tokenized_problem[:, :cur],
                                               number_tensors[num_count],
                                               tokenized_problem[:, cur + len(quant_list_ids):]], dim=1)

                cur += len(number_tensors[num_count][0]) - len(quant_list_ids)
                num_count += 1
            cur += 1

        return tokenized_problem, question_mask, number_mask, num_count

    def _num2quent(self, problem_text: str):
        # 문제에 등장하는 모든 number변수를 " <quant> "로 치환

        append_idx = 0
        for find_number in re.finditer("number\d+", problem_text):
            if find_number.start() == 0:
                problem_text = " " + problem_text
                append_idx += 1

            if find_number.end() + append_idx >= len(problem_text):
                problem_text = problem_text + " "

            l_space = "" if problem_text[find_number.start() + append_idx - 1] == " " else " "
            r_space = "" if problem_text[find_number.end() + append_idx] == " " else " "
            problem_text = problem_text[:find_number.start() + append_idx] + \
                           l_space + "<quant>" + r_space + problem_text[find_number.end() + append_idx:]

            append_idx = append_idx + len("<quant>") - len(find_number.group())

        return problem_text

    def __getitem__(self, index) -> Feature:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)
