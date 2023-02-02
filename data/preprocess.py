import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

BASE_PATH = '../'


class Equation:
    def __init__(self, raw: Optional[str] = None, type: str = "prefix"):
        self.equation = None

        if raw is not None:
            if type == "prefix":
                self.equation = self.prefix2equation(raw)
            elif type == "formula":
                self.equation = self.formular2eqation(raw)

    def getList(self) -> Optional[list[list[str]]]:
        return self.equation

    def getOperator(self) -> Optional[list[str]]:
        if self.equation is None:
            return None

        operator = []
        for e in self.equation:
            operator.append(e[0])

        return operator

    def getArgument(self) -> Optional[list[list[str]]]:
        if self.equation is None:
            return None

        argument = []
        for e in self.equation:
            argument.append(e[1:])

        return argument

    def formular2eqation(self, formula: str) -> list[list[str]]:
        equation = []
        formula = formula.strip("|")  # | 이 마지막에 있는 경우도 있고 없는 경우도 있으므로
        formula = formula.replace("(", ",")  # 괄호를 ,로 바꿔서 ,로 split 하자
        formula = formula.replace(")", "")
        formula = formula.split("|")

        for f in formula:
            entities = f.split(",")
            equation.append(entities)

        return equation

    def prefix2equation(self, prefix: str) -> list[list[str]]:
        @dataclass
        class Operator:
            data: str

        @dataclass
        class Operand:
            data : str

        def checkLeaf(cur: int, prefix: list[str]) -> bool:
            if cur + 2 >= len(prefix):
                return False

            if type(prefix[cur]) == Operator and type(prefix[cur + 1]) == Operand and type(prefix[cur+2]) == Operand:
                return True
            else:
                return False

        operator_dict = {
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
        }

        equation = []

        # 1. operator를 구분해야 함. operator의 종류 : +, -, *, /
        # 2. operend를 구분해야 함. operend의 종류 : #0, #1, ... n1, n2, ... 100.0(숫자) 등
        prefix = prefix.replace("number", "n")

        prefix_list = []
        for p in prefix.split(" "):
            if re.match(r"[+\-/*]", p):
                prefix_list.append(Operator(p))
            else:
                prefix_list.append(Operand(p))

        # (operator, operand, operand) => #number
        result_cnt = 0
        while len(prefix_list) != 1:
            temp = []
            cur = 0
            while cur < len(prefix_list):
                if checkLeaf(cur, prefix_list):
                    equation.append([operator_dict[prefix_list[cur].data], prefix_list[cur+1].data, prefix_list[cur+2].data])
                    temp.append(Operand("#" + str(result_cnt)))
                    result_cnt += 1
                    cur += 2
                else:
                    temp.append(prefix_list[cur])
                cur += 1

            prefix_list = temp

        return equation


class Problem:
    def __init__(self, problem: str, numbers: list[str], equation: Equation):
        self.context = None
        self.question = None
        self.numbers = numbers
        self.equation = equation.getList()
        self.golden_op = equation.getOperator()
        self.golden_argument = equation.getArgument()

        problem = self.toNumProblem(problem)
        self.context, self.question = problem2CQ(problem)

    def toNumProblem(self, problem: str) -> str:
        for i, n in enumerate(self.numbers):
            problem = problem.replace(n, f"number{i}")

        return problem

    def __repr__(self):
        return f"Problem(context={self.context}, question={self.question}, " \
               f"numbers={self.numbers}, equation={self.equation}, " \
               f"golden_op={self.golden_op}, golden_argument={self.golden_argument})"

#json serialization class (for json.dump)
class ProblemEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Problem):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

def problem2CQ(problem : str) -> Tuple[str, str]:
    sentences = problem.strip().split(".")
    context, question = ".".join(sentences[:-1]) + ".", sentences[-1].strip()

    return context, question

# 문제에 등장한 문자열 그대로 추출하는 함수 => 따라서 후처리를 통해 숫자만 추출해야할 필요가 생길 수 있음
def extractNum(problem : str):
    # 문제에 등장하는 숫자의 종류
    # 숫자 종류: 10000 (자연수), 1,000,000 (쉼표가 있는 숫자), 1.5 (소수점이 있는 숫자), - 4 or -4(부호가 있는 숫자)
    numbers = re.findall(r'(?:[-+][ ]?)?\d+(?:\.\d+|(?:,\d\d\d)+)?', problem)
    return numbers


#mathqa preprocessing
def preprocess_mathqa(file_path : str = "data/raw/mathqa", save_path : str = "data/processed/mathqa"):
    train_path = Path(BASE_PATH, file_path, "train.json")
    dev_path = Path(BASE_PATH, file_path, "dev.json")
    test_path = Path(BASE_PATH, file_path, "test.json")

    dataset_path = [train_path, dev_path, test_path]

    for path in dataset_path:
        print(f"preprocessing {path}...")
        with open(path, 'r') as f:
            problem_list = []

            data = json.load(f)
            print(f"number of problems: {len(data)}")
            for problem in data:
                problem_text, problem["Problem"]
                numbers = extractNum(problem["Problem"])
                equation = Equation(problem["linear_formula"])

                problem = Problem(problem_text, numbers, equation)
                problem_list.append(problem)

        processed_path = Path(BASE_PATH, save_path, f"{path.stem}_preprocessed.json")

        if not os.path.exists(processed_path.parent):
            os.makedirs(processed_path.parent)
        with open(processed_path, 'w') as f:
            json.dump(problem_list, f, indent=4, cls=ProblemEncoder)


#svamp preprocessing
def preprocess_svamp(file_path : str = "data/raw/mawps-asdiv-a_svamp", save_path : str = "data/processed/svmap"):
    train_path = Path(BASE_PATH, file_path, "train.csv")
    dev_path = Path(BASE_PATH, file_path, "dev.csv")

    dataset_path = [train_path, dev_path]

    for path in dataset_path:
        print(f"preprocessing {path}...")
        data = pd.read_csv(path)

        problem_list = []

        print(f"number of problems: {len(data)}")
        for problem in data.itertuples():
            problem_text = problem.Qutestino
            numbers = problem.Numbers
            equation = Equation(problem.Equation, type="prefix")

            problem = Problem(problem_text, numbers, equation)
            problem_list.append(problem)

        processed_path = Path(BASE_PATH, save_path, f"{path.stem}_preprocessed.json")

        if not os.path.exists(processed_path.parent):
            os.makedirs(processed_path.parent)
        with open(processed_path, 'w') as f:
            json.dump(problem_list, f, indent=4, cls=ProblemEncoder)



if __name__ == "__main__":
    preprocess_mathqa("data/raw/mathqa", "data/processed/mathqa")
    preprocess_svamp("data/raw/mawps-asdiv-a_svamp", "data/processed/svmap")