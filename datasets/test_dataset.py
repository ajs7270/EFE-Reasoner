from unittest import TestCase

from transformers import AutoTokenizer
import re
from dataset import Dataset, Problem, Feature
import torch


class TestDataset(TestCase):
    tokenizers = []
    quant_list_ids = []
    models = ["roberta-base", "roberta-large"] #, "microsoft/deberta-v3-base", "microsoft/deberta-base"]
    for model_name in models:
        tokenizers.append(AutoTokenizer.from_pretrained(model_name))
    for tokenizer in tokenizers:
        quant_list_ids.append(tokenizer(" <quant> ", return_tensors="pt").input_ids[0][1:-2])

    def _num2quant(self, problem_text: str):
        # before tokenization, change all numbers to token "<quant>"
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

    def refine_test_translate2number(self, problem, tokenizer, i):
        # get tokenized numbers using number_mask, append to num_list
        problem_question = self._num2quant(problem.question)
        problem_context = self._num2quant(problem.context)
        # tokenize
        tokenized_problem = self.tokenizer(problem_context, problem_question, return_tensors="pt").input_ids
        tokenized_context = self.tokenizer(problem_context, return_tensors="pt").input_ids
        # first token "<s>", last token "</s>" is removed
        number_tensors = [self.tokenizer(number, return_tensors="pt").input_ids[:, 1:-1] for number in
                          problem.numbers]

        tokenized_problem, question_mask, number_mask, num_count = Dataset._translate2number(tokenized_problem,
                                                                                             tokenized_context,
                                                                                             number_tensors,
                                                                                             self.quant_list_ids[i])
        num_list = []
        decoded_sentence = ""
        tokenized_question = ""
        # get tokens of cur_num, decode them and append to num_list
        for cur_num in range(max(number_mask[0])):
            mask = number_mask[0] == (cur_num + 1)
            tokenized_num = torch.masked_select(tokenized_problem[0], mask)
            num_list.append(tokenizer.decode(tokenized_num))
        # strip whitespaces from problem and decoded tokenized_problem, also removed "<s>", "</s>"
        for k in range(len(tokenized_problem[0])):
            decoded_sentence += tokenizer.decode(tokenized_problem[0][k])
        decoded_sentence = decoded_sentence.replace("<s>", "").replace("</s>", "").replace(" ", "")
        problem_sentence = problem.context + problem.question
        for k, num in enumerate(num_list):
            problem_sentence = problem_sentence.replace(f"number{k}", num).replace(" ", "")
        # strip whitespaces from question and decoded tokenized question, also removed "<s>", "</s>"
        for k in range(len(question_mask[0])):
            if question_mask[0][k] != 0:
                tokenized_question += tokenizer.decode(tokenized_problem[0][k])
        tokenized_question = tokenized_question.replace(" ", "").replace("</s>", "")
        question_sentence = problem.question
        for k, num in enumerate(num_list):
            question_sentence = question_sentence.replace(f"number{k}", num).replace(" ", "")
        return num_count, num_list, decoded_sentence, problem_sentence, tokenized_question, question_sentence

    def test_translate2number(self):
        print(f"check models: ", end="")
        for i, tokenizer in enumerate(self.tokenizers):
            print("\033[32m" + self.models[i] + "\033[0m", end=", ")
            problem_dict1 = {
                'context': 'sophia finished number0 / number1 of a book . she calculated that she finished number2 more pages than she has yet to read .',
                'question': 'how long is her book ?',
                'numbers': ['23838', '32131313', '90'],
                'equation': [['divide', 'n0', 'n1'], ['subtract', 'const_1', '#0'], ['divide', 'n2', '#1']],
                'golden_op': ['divide', 'subtract', 'divide'],
                'golden_argument': [['n0', 'n1'], ['const_1', '#0'], ['n2', '#1']],
                'same_number_idx': [[0, 1], [2]],
            }
            problem1 = Problem(**problem_dict1)
            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(problem1, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem1.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem1.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict2 = {
                "context": "a mixture of number0 liters of milk and water contains number1 % water .",
                "question": "how much water should be added to this so that water may be number2 % in the new mixture ?",
                "numbers": ["40", "10", "20"],
                "same_number_idx": [],
                "equation": [["divide", "n2", "const_100"], ["divide", "n1", "const_100"]],
                "golden_op": ["divide", "divide", "subtract", "divide", "multiply", "multiply", "subtract", "divide"],
                "golden_argument": [["n2", "const_100"], ["n1", "const_100"], ["const_100", "n2"], ["#2", "const_100"]]
            }
            problem2 = Problem(**problem_dict2)

            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(
                problem2, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem2.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem2.numbers)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_problem, original_problem)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_question, original_question)

            problem_dict3 = {
                "context": "in a certain lottery , the probability that a number between number0 and number1 , "
                           "inclusive , is drawn is number2 / number3 .",
                "question": "if the probability that a number number4 or larger is drawn is number5 / number6 , what is the probability that a number less than or equal to number7 is drawn ?",
                "numbers": ["14", "20", "1434", "6", "1414414", "2", "3", "20"],
                "same_number_idx": [[0, 1], [2, 3], [4, 5, 6], [7]],
                "equation": [["divide", "n2", "n3"], ["divide", "n5", "n6"], ["subtract", "const_1", "#0"],
                             ["divide", "n4", "n1"], ["subtract", "const_1", "#1"], ["divide", "n0", "n1"],
                             ["subtract", "const_1", "#2"], ["divide", "n7", "n1"]],
                "golden_op": ["divide", "divide", "subtract", "divide", "subtract", "divide", "subtract", "divide"],
                "golden_argument": [["n2", "n3"], ["n5", "n6"], ["const_1", "#0"], ["n4", "n1"], ["const_1", "#1"],
                                    ["n0", "n1"], ["const_1", "#2"], ["n7", "n1"]]
            }
            problem3 = Problem(**problem_dict3)

            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(
                problem3, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem3.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem3.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict4 = {
                "context": "i . x + number0 y + number1 z = number2 ii . x + y - z = number3 iii .",
                "question": "number4 x + number5 y - z = number6 what is the value of y in the system above ?",
                "numbers": ["234", "31444", "772", "055", "0", "2", "1"],
                "same_number_idx": [[0, 1], [2, 3], [4, 5, 6]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "n3"], ["divide", "n4", "n5"],
                             ["divide", "n6", "n7"], ["subtract", "const_1", "#0"], ["subtract", "const_1", "#1"],
                             ["subtract", "const_1", "#2"], ["subtract", "const_1", "#3"], ["divide", "n0", "n1"],
                             ["divide", "n2", "n3"], ["divide", "n4", "n5"], ["divide", "n6", "n7"]],
                "golden_op": ["divide", "divide", "divide", "divide", "subtract", "subtract", "subtract", "subtract",
                              "divide", "divide", "divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "n3"], ["n4", "n5"], ["n6", "n7"], ["const_1", "#0"],
                                    ["const_1", "#1"], ["const_1", "#2"], ["const_1", "#3"], ["n0", "n1"], ["n2", "n3"],
                                    ["n4", "n5"], ["n6", "n7"]]
            }

            problem4 = Problem(**problem_dict4)

            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(
                problem4, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem4.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem4.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict5 = {
                "context": ".",
                "question": "if grapes are number0 % water and raisins are number1 % water , then how many kilograms did a quantity of raisins , which currently weighs number2 kilograms , weigh when all the raisins were grapes ? ( assume that the only difference between their raisin - weight and their grape - weight is water that evaporated during their transformation . )",
                "numbers": ["90", "233333330", "10"],
                "same_number_idx": [[0, 1], [2]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "const_100"]],
                "golden_op": ["divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "const_100"]]
            }
            problem5 = Problem(**problem_dict5)
            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(
                problem5, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem5.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem5.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)

            problem_dict6 = {
                "context": "the volume of a sphere with radius r is ( number0 / number1 ) * pi * r ^ number2 and the surface area is number3 * pi * r ^ number4 .",
                "question": "if a sperical balloon has a volume of number5 pi cubic centimeters , what is hte surface area of the balloon in square centimeters ?",
                "numbers": ["4", "3", "3", "4", "3", "12345"],
                "same_number_idx": [[0, 1], [2, 3, 4], [5]],
                "equation": [["divide", "n0", "n1"], ["divide", "n2", "const_100"], ["divide", "n3", "const_100"],
                             ["divide", "n4", "const_100"], ["divide", "n5", "const_100"]],
                "golden_op": ["divide", "divide", "divide", "divide", "divide"],
                "golden_argument": [["n0", "n1"], ["n2", "const_100"], ["n3", "const_100"], ["n4", "const_100"],
                                    ["n5", "const_100"]]
            }
            problem6 = Problem(**problem_dict6)
            num_count, decoded_num_list, decoded_problem, original_problem, decoded_question, original_question = self.refine_test_translate2number(
                problem6, tokenizer, i)
            # check whether num_count equals length of problem.numbers
            self.assertEqual(num_count, len(problem6.numbers))
            # check equality of decoded tokenized numbers and problem.numbers
            self.assertEqual(decoded_num_list, problem6.numbers)
            # check equality of decoded sentence and original problem sentence
            self.assertEqual(decoded_problem, original_problem)
            # check whether question_mask is correct after stripping whitespaces
            self.assertEqual(decoded_question, original_question)