from unittest import TestCase
from preprocess import extractNum, problem2CQ, Equation


class ProblemTest(TestCase):
    def test_problem2cq(self):
        # mathqa sample
        problem1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 2 of the 6 workers will be randomly chosen to be interviewed . what is the probability that joshua and jose will both be chosen ?"
        context1 = "oshua and jose work at an auto repair center with 3 other workers . for a survey on health care insurance , 2 of the 6 workers will be randomly chosen to be interviewed ."
        question1 = "what is the probability that joshua and jose will both be chosen ?"
        self.assertEqual(problem2CQ(problem1), (context1, question1))

        # svamp sample
        problem2 = "every day ryan spends number0 hours on learning english and some more hours on learning chinese . if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?"
        context2 = "every day ryan spends number0 hours on learning english and some more hours on learning chinese ."
        question2 = "if he spends number1 hours more on learning english than on learning chinese how many hours does he spend on learning chinese ?"
        self.assertEqual(problem2CQ(problem2), (context2, question2))

        # mawps and asdiv sample
        problem3 = "A chef needs to cook number0 potatoes . He has already cooked number1 . If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        context3 = "A chef needs to cook number0 potatoes . He has already cooked number1 ."
        question3 = "If each potato takes number2 minutes to cook , how long will it take him to cook the rest ?"
        self.assertEqual(problem2CQ(problem3), (context3, question3))

    def test_extract_num(self):
        # 숫자 종류: 10000 (자연수), 1,000,000 (쉼표가 있는 숫자), 1.5 (소수점이 있는 숫자), - 4 or -4(부호가 있는 숫자)
        self.assertEqual(extractNum("es - 4 ≤ x ≤ 5 and"), ["- 4", "5"])
        self.assertEqual(extractNum("es -4 ≤ x ≤ 5 and"), ["-4", "5"])
        self.assertEqual(extractNum("and 6 ≤ y ≤ 16 . -4 ho"), ["6", "16", "-4"])
        self.assertEqual(extractNum("atest 6 - digit  divided by 6 , 7 , 8 , 9 ,"), ["6", "6", "7", "8", "9"])
        self.assertEqual(extractNum(" sum of a number and its square is 20 , what i"), ["20"])
        self.assertEqual(extractNum("d $ 5,000 to open -123 - 123"), ["5,000", "-123", "- 123"])
        self.assertEqual(extractNum("and 6 ≤ y ≤ 16 . -4 ho"), ["6", "16", "-4"])


class TestEquation(TestCase):
    def test_formular2eqation(self):
        linear_formula1 = "add(n1,const_1)|"
        equation1 = Equation(linear_formula1, type="formula")
        self.assertEqual(equation1.getList(), [["add", "n1", "const_1"]])

        linear_formula2 = "multiply(n0,n1)|divide(n2,#0)"
        equation2 = Equation(linear_formula2, type="formula")
        self.assertEqual(equation2.getList(), [["multiply", "n0", "n1"], ["divide", "n2", "#0"]])

        linear_formula3 = "divide(n0,n1)|multiply(const_1,const_1000)|divide(#1,#0)|subtract(#2,n1)|"
        equation3 = Equation(linear_formula3, type="formula")
        self.assertEqual(equation3.getList(),
                         [["divide", "n0", "n1"], ["multiply", "const_1", "const_1000"], ["divide", "#1", "#0"],
                          ["subtract", "#2", "n1"]])

        linear_formula4 = "add(n1,n2)|add(n3,#0)|subtract(const_100,#1)|multiply(n0,#2)|divide(#3,const_100)"
        equation4 = Equation(linear_formula4, type="formula")
        self.assertEqual(equation4.getList(),
                         [["add", "n1", "n2"], ["add", "n3", "#0"], ["subtract", "const_100", "#1"],
                          ["multiply", "n0", "#2"], ["divide", "#3", "const_100"]])

    def test_get_operator_argument(self):
        linear_formula1 = "add(n1,const_1)|"
        equation1 = Equation(linear_formula1, type="formula")
        self.assertEqual(equation1.getOperator(), ["add"])
        self.assertEqual(equation1.getArgument(), [["n1", "const_1"]])

        linear_formula2 = "multiply(n0,n1)|divide(n2,#0)"
        equation2 = Equation(linear_formula2, type="formula")
        self.assertEqual(equation2.getOperator(), ["multiply", "divide"])
        self.assertEqual(equation2.getArgument(), [["n0", "n1"], ["n2", "#0"]])

        linear_formula3 = "divide(n0,n1)|multiply(const_1,const_1000)|divide(#1,#0)|subtract(#2,n1)|"
        equation3 = Equation(linear_formula3, type="formula")
        self.assertEqual(equation3.getOperator(), ["divide", "multiply", "divide", "subtract"])
        self.assertEqual(equation3.getArgument(), [["n0", "n1"], ["const_1", "const_1000"], ["#1", "#0"], ["#2", "n1"]])

        linear_formula4 = "add(n1,n2)|add(n3,#0)|subtract(const_100,#1)|multiply(n0,#2)|divide(#3,const_100)"
        equation4 = Equation(linear_formula4, type="formula")
        self.assertEqual(equation4.getOperator(), ["add", "add", "subtract", "multiply", "divide"])
        self.assertEqual(equation4.getArgument(),
                         [["n1", "n2"], ["n3", "#0"], ["const_100", "#1"], ["n0", "#2"], ["#3", "const_100"]])

    def test_prefix2equation(self):
        converter = Equation()

        prefix1 = "* number0 number1"
        self.assertEqual(converter.prefix2equation(prefix1), [["multiply", "n0", "n1"]])

        prefix2 = "* number0 + number1 number2"
        self.assertEqual(converter.prefix2equation(prefix2), [["add", "n1", "n2"], ["multiply", "n0", "#0"]])

        prefix3 = "+ + number0 number1 number2"
        self.assertEqual(converter.prefix2equation(prefix3), [["add", "n0", "n1"], ["add", "#0", "n2"]])

        prefix4 = "* / - number1 number0 number0 100.0"
        self.assertEqual(converter.prefix2equation(prefix4), [["subtract", "n1", "n0"], ["divide", "#0", "n0"], ["multiply", "#1", "100.0"]])
