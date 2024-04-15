# Syntax specificaton

> [!WARNING]
> **THIS IS UNOFFICIAL** and only for filskalang implementation!
> See [the original implementation](https://github.com/rkneusel9/StrangeCodeBook/blob/master/chapter_12/filska.py) to check the real syntax.

Syntax using [Wirth syntax notation](https://en.wikipedia.org/wiki/Wirth_syntax_notation):

```
Syntax = { SubProgram } .
SubProgram = "{" identifier { Instruction } "}" .
Instruction = NullaryInstruction | UnaryInstruction | BinaryInstruction | JumpInstruction | TestInstruction
NullaryInstruction = nullary_operator
UnaryInstruction = unary_operator "," int_literal
BinaryInstruction = AssignmentInstruction | SwapInstruction
JumpInstruction = jump_operator "," identifier
TestInstruction = test_operator "," test_flag "," int_literal
AssignmentInstruction = assignment_operator "," Assignment
Assignment = register "=" register register
SwapInstruction = swap_operator "," register register


nullary_operator = "inc" | "dec" | "sin" | "cos" | "tan" | "log" | "exp" | "flr" | "cel" | "rnd" | "neg" | "chr" | "prt" | "ipt" | "sqr" | "asn" | "acs" | "atn" | "tmx" | "tmy" | "txm" | "tym" | "tzm" | "tmz" | "hlt" .
unary_operator = "set" | "gto" | "cmp"
assignment_operator = "add" | "sub" | "mul" | "div" | "mod" | "pow"
swap_operator = "swp"
jump_operator = "jmp" | "jpr"
register = "x" | "y" | "z" | "m"
test_flag = "z" | "e" | "l" | "g" | "n"
```
