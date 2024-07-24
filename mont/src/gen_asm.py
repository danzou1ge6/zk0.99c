from dataclasses import dataclass
from typing import TypeVar, Callable, Optional
from pathlib import Path

@dataclass
class Asm:
    inst: list[str]
    outputs: list[str]
    inputs: list[str]

    def to_lines(self, volatile = True) -> list[str]:
        s = []
        s.append("asm " + ("volatile" if volatile else "") + "(")
        s.append(f'  "{{\\n\\t"')
        s += (f"  \"{line};\\n\\t\"" for line in self.inst)
        s.append(f'  "}}"')
        s.append("  : " + ", ".join(self.outputs))
        s.append("  : " + ", ".join(self.inputs))
        s.append(");")
        return s
    
    def to_str(self, volatile = True) -> str:
        return "\n".join(self.to_lines(volatile=volatile))


def montgomery_reduction(n_words: int) -> Asm:
    input_a = lambda i: f"a.c{i}"
    input_m = lambda i: f"m.c{i}"
    input_m_prime = "m_prime"

    reg_a = lambda i: f"%{i}"
    reg_m = lambda i: f"%{i + 2 * n_words}"
    reg_m_prime = f"%{3 * n_words}"
    reg_u_i = "t1"
    reg_carry2 = "t2"
    reg_temp = "t3"

    inst = [
        '.reg .u32 t1',
        '.reg .u32 t2',
        '.reg .u32 t3',
    ]

    # carry2 begins being 0
    inst.append(f"mov.u32 {reg_carry2}, 0")

    for i in range(n_words):
        # u_i <- a_i m ' mod b where b = 2^32
        inst.append(f"mul.lo.u32 {reg_u_i}, {reg_a(i)}, {reg_m_prime}")

        # Add low 32 bits of u_i m  b^i to A
        inst.append(f"mad.lo.cc.u32 {reg_a(i)}, {reg_u_i}, {reg_m(0)}, {reg_a(i)}")
        for j in range(1, n_words):
            inst.append(f"madc.lo.cc.u32 {reg_a(i + j)}, {reg_u_i}, {reg_m(j)}, {reg_a(i + j)}")
        # Add carry bit to carry2, which will be later added to a_(i + n_words)
        inst.append(f"addc.u32 {reg_carry2}, {reg_carry2}, 0")

        # Add high 32 bits of u_i m  b^i to A
        inst.append(f"mad.hi.cc.u32 {reg_a(i + 1)}, {reg_u_i}, {reg_m(0)}, {reg_a(i + 1)}")
        for j in range(1, n_words):
            inst.append(f"madc.hi.cc.u32 {reg_a(i + j + 1)}, {reg_u_i}, {reg_m(j)}, {reg_a(i + j + 1)}")
        if i != n_words - 1:
            # Add carry bit to tmp; Final i iteration doesn't need this.
            inst.append(f"addc.u32 {reg_temp}, 0, 0")
        # Add carry2 from last i iteration to a_(i + n_words)
        inst.append(f"add.cc.u32 {reg_a(i + n_words)}, {reg_a(i + n_words)}, {reg_carry2}")
        if i != n_words - 1:
            # Add carry bit to tmp, which is the new carry2
            inst.append(f"addc.u32 {reg_carry2}, {reg_temp}, 0")
    
    return Asm(
        inst = inst,
        outputs = [f'"+r"({input_a(i)})' for i in range(2 * n_words)],
        inputs = [f'"r"({input_m(i)})' for i in range(n_words)] + [f'"r"({input_m_prime})']
    )
    

def mul(n_words):
    input_a = lambda i: f"c{i}"
    input_b = lambda i: f"rhs.c{i}"
    output_r = lambda i: f"r.c{i}"

    reg_r = lambda i: f"%{i}"
    reg_a = lambda i: f"%{i + 2 * n_words}"
    reg_b = lambda i: f"%{i + 3 * n_words}"

    inst = []

    # First calculate a_0 B
    # Low 32 bits
    for j in range(n_words):
        inst.append(f"mul.lo.u32 {reg_r(j)}, {reg_a(0)}, {reg_b(j)}")
    # High 32 bits
    inst.append(f"mad.hi.cc.u32 {reg_r(1)}, {reg_a(0)}, {reg_b(0)}, {reg_r(1)}")
    for j in range(1, n_words - 1):
        inst.append(f"madc.hi.cc.u32 {reg_r(j + 1)}, {reg_a(0)}, {reg_b(j)}, {reg_r(j + 1)}")
    inst.append(f"madc.hi.cc.u32 {reg_r(n_words)}, {reg_a(0)}, {reg_b(n_words - 1)}, 0")
    # Add carry
    inst.append(f"addc.u32 {reg_r(n_words + 1)}, 0, 0")

    for i in range(1, n_words):
        # Add low 32 bits of a_i B b^i to R
        inst.append(f"mad.lo.cc.u32 {reg_r(i)}, {reg_a(i)}, {reg_b(0)}, {reg_r(i)}")
        for j in range(1, n_words):
            inst.append(f"madc.lo.cc.u32 {reg_r(i + j)}, {reg_a(i)}, {reg_b(j)}, {reg_r(i + j)}")
        # Add carry to a_(i + n_words). This can't overflow as a_(i + n_words) is at most 1.
        inst.append(f"addc.u32 {reg_r(i + n_words)}, {reg_r(i + n_words)}, 0")

        # Add high 32 bits of a_i B b^i to R
        inst.append(f"mad.hi.cc.u32 {reg_r(i + 1)}, {reg_a(i)}, {reg_b(0)}, {reg_r(i + 1)}")
        for j in range(1, n_words):
            inst.append(f"madc.hi.cc.u32 {reg_r(i + j + 1)}, {reg_a(i)}, {reg_b(j)}, {reg_r(i + j + 1)}")
        # Add carry to a_(i + n_words + 1), if index i + n_words + 1 in range.
        if i != n_words - 1:
            inst.append(f"addc.u32 {reg_r(i + n_words + 1)}, 0, 0")
    
    return Asm(
        inst = inst,
        outputs = [f'"=r"({output_r(i)})' for i in range(2 * n_words)],
        inputs = [f'"r"({input_a(i)})' for i in range(n_words)] + [f'"r"({input_b(i)})' for i in range(n_words)]
    )

T = TypeVar("T")
def index_of_predicate(lst: list[T], predicate: Callable[[T], bool], begin: int = 0) -> Optional[int]:
    for i, item in enumerate(lst):
        if i >= begin and predicate(item):
            return i
    return None
    
def count_indentation(s: str) -> int:
    cnt = 0
    while s[cnt] == ' ':
        cnt += 1
    return cnt

def insert_generated(lines: list[str], point_name: str, to_insert: list[str]) -> list[str]:
    i = index_of_predicate(lines, lambda line: line.lstrip().rstrip() == f"// >>> GENERATED: {point_name}")
    j = index_of_predicate(lines, lambda line: line.lstrip().rstrip() == "// >>> GENERATED END", begin = i)
    if i is None or j is None:
        raise RuntimeError(f"Point {point_name} not found in document")
    indent = count_indentation(lines[i])
    r = lines[:i + 1]
    r += (indent * " " + line for line in to_insert)
    r += lines[j:]
    return r

from sys import argv

if __name__ == "__main__":
    if len(argv) != 3:
        raise RuntimeError("Need two arguments, for input file and output file")
    src = argv[1]
    target = argv[2]
    if not Path(src).exists():
        raise FileNotFoundError(src)

    with open(src, "r") as rf:
        code_txt = rf.read()
    code_lines = code_txt.split("\n")

    code_lines = insert_generated(
        code_lines,
        "montgomery_reduction",
        montgomery_reduction(8).to_lines()
    )
    code_lines = insert_generated(
        code_lines,
        "mul",
        mul(8).to_lines()
    )

    with open(target, "w") as wf:
        wf.write("\n".join(code_lines))

