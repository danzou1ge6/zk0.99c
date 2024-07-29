from dataclasses import dataclass
from typing import TypeVar, Callable, Optional, List
from pathlib import Path
from math import ceil

@dataclass
class Asm:
    inst: List[str]
    outputs: List[str]
    inputs: List[str]

    def to_lines(self, volatile = True) -> List[str]:
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

def square(n_words: int) -> Asm:
    @dataclass
    class Cell:
        hi: bool
        a: int
        b: int

        def op(self, cc: bool, c: bool) -> str:
            op = 'mad'
            if c:
                op += 'c'
            if self.hi:
                op += '.hi'
            else:
                op += '.lo'
            if cc:
                op += '.cc'
            op += '.u32'
            return op
        
        def r_i(self) -> int:
            return self.a + self.b + (1 if self.hi else 0)

    def cells_of_col(i_col: int) -> List[Cell]:
        cells = []
        for a in range(0, ceil((i_col - 1)/ 2)):
            b = i_col - 1 - a
            if 0 <= a < 8 and 0 <= b < 8:
                cells.append(Cell(hi=True, a=a, b=b))
        for a in range(0, ceil(i_col / 2)):
            b = i_col - a
            if 0 <= a < 8 and 0 <= b < 8:
                cells.append(Cell(hi=False, a=a, b=b))
        return cells

    def plan_carries(cols: List[List[Cell]]) -> List[List[Cell]]:
        scans = []
        for end_i_col in range(2, 15):
            cells = []
            for i_col in range(end_i_col, 0, -1):
                if len(cols[i_col]) != 0:
                    cells.insert(0, cols[i_col].pop(0))
                else:
                    break
            scans.append(cells)
        return scans
    
    cols = [cells_of_col(i) for i in range(2 * n_words - 1)]
    # A series of mads, beginning with mad.cc and ends with madc, with madc.cc in between
    scans = plan_carries(cols)

    input_a = lambda i: f"c{i}"
    output_r = lambda i: f"r.c{i}"

    reg_a = lambda i: f"%{i + 2 * n_words}"
    r_defined = [False for _ in range(n_words * 2)]
    reg_r_read = lambda i: f"%{i}" if r_defined[i] else "0"
    def reg_r_write(i: int):
        r_defined[i] = True
        return f"%{i}"
    reg_tmp = "t1"

    inst = [
        ".reg .u32 t1",
    ]

    # Calculates sum_(i != j) b^(i + j) a_i a_j, storing in R = (r15, ..., r0)
    for scan in scans:
        for i, cell in enumerate(scan):
            op = cell.op(i != len(scan) - 1, i != 0)
            i_r = cell.r_i()
            reg_read = reg_r_read(i_r)
            inst.append(f"{op} {reg_r_write(i_r)}, {reg_a(cell.a)}, {reg_a(cell.b)}, {reg_read}")

    # Calculates 2 * sum_(i != j) b^(i + j) a_i a_j, storing in R
    inst.append(f"shr.b32 {reg_r_write(2 * n_words - 1)}, {reg_r_read(2 * n_words - 2)}, 31")
    for i in reversed(range(2, 2 * n_words - 1)):
        inst.append(f"shl.b32 {reg_r_write(i)}, {reg_r_read(i)}, 1")
        inst.append(f"shr.b32 {reg_tmp}, {reg_r_read(i - 1)}, 31")
        inst.append(f"or.b32 {reg_r_write(i)}, {reg_tmp}, {reg_r_read(i)}")
    inst.append(f"shl.b32 {reg_r_write(1)}, {reg_r_read(1)}, 1")

    # Calculates sum_i b^(2i) a_i^2, storing in R
    reg_read = reg_r_read(0)
    inst.append(f"mad.lo.cc.u32 {reg_r_write(0)}, {reg_a(0)}, {reg_a(0)}, {reg_read}")
    inst.append(f"mad.hi.cc.u32 {reg_r_write(1)}, {reg_a(0)}, {reg_a(0)}, {reg_r_read(1)}")
    for i in range(1, n_words):
        inst.append(f"madc.lo.cc.u32 {reg_r_write(2 * i)}, {reg_a(i)}, {reg_a(i)}, {reg_r_read(2 * i)}")
        inst.append(f"madc.hi.cc.u32 {reg_r_write(2 * i + 1)}, {reg_a(i)}, {reg_a(i)}, {reg_r_read(2 * i + 1)}")
    
    return Asm(
        inst = inst,
        outputs = [f'"=r"({output_r(i)})' for i in range(2 * n_words)],
        inputs = [f'"r"({input_a(i)})' for i in range(n_words)]
    )


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
    

def mul(n_words: int) -> Asm:
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

def host_mul(n_words: int) -> List[str]:
    var_r = lambda i: f"r{i}"
    var_carry = "carry"
    var_lhs = lambda i: f"c{i}"
    var_rhs = lambda i: f"rhs.c{i}"

    lines = []

    lines.append(f"u32 {', '.join((var_r(i) for i in range(2 * n_words)))};")

    lines.append(f"u32 {var_carry} = 0;")
    for j in range(0, n_words):
        lines.append(f"{var_r(j)} = madc(0, {var_lhs(0)}, {var_rhs(j)}, {var_carry});")
    lines.append(f"{var_r(n_words)} = {var_carry};")

    for i in range(1, n_words):
        lines.append(f"{var_carry} = 0;")
        for j in range(0, n_words):
            lines.append(f"{var_r(i + j)} = madc({var_r(i + j)}, {var_lhs(i)}, {var_rhs(j)}, {var_carry});")
        lines.append(f"{var_r(i + n_words)} = {var_carry};")
    
    lines.append(f"Number2 r({', '.join((var_r(i) for i in range(2 * n_words)))});")
    lines.append(f"return r;")
    
    return lines

def host_square(n_words: int) -> List[str]:
    var_r = lambda i: f"r{i}"
    var_carry = f"carry"
    var_c = lambda i: f"c{i}"

    lines = []

    lines.append(f"u32 {', '.join((var_r(i) for i in range(2 * n_words)))};")

    lines.append(f"u32 {var_carry} = 0;")
    for j in range(1, n_words):
        lines.append(f"{var_r(j)} = madc(0, {var_c(0)}, {var_c(j)}, {var_carry});")
    lines.append(f"{var_r(n_words)} = {var_carry};")

    for i in range(1, n_words - 1):
        lines.append(f"{var_carry} = 0;")
        for j in range(i + 1, n_words):
            lines.append(f"{var_r(i + j)} = madc({var_r(i + j)}, {var_c(i)}, {var_c(j)}, {var_carry});")
        lines.append(f"{var_r(i + n_words)} = {var_carry};")
    
    lines.append(f"{var_r(2 * n_words - 1)} = {var_r(2 * n_words - 2)} >> 31;")
    for i in reversed(range(2, 2 * n_words - 1)):
        lines.append(f"{var_r(i)} = ({var_r(i)} << 1) | ({var_r(i - 1)} >> 31);")
    lines.append(f"{var_r(1)} = {var_r(1)} << 1;")

    lines.append(f"{var_carry} = 0;")
    lines.append(f"{var_r(0)} = madc(0, {var_c(0)}, {var_c(0)}, {var_carry});")
    lines.append(f"{var_r(1)} = addc(0, {var_r(1)}, {var_carry});")
    for i in range(2, n_words * 2, 2):
        lines.append(f"{var_r(i)} = madc({var_r(i)}, {var_c(int(i / 2))}, {var_c(int(i / 2))}, {var_carry});")
        lines.append(f"{var_r(i + 1)} = addc(0, {var_r(i + 1)}, {var_carry});")
    
    lines.append(f"Number2 r({', '.join(var_r(i) for i in range(2 * n_words))});")
    lines.append("return r;")

    return lines

def host_montgomery_reduction(n_words: int) -> List[str]:
    var_r = lambda i: f"a.c{i}"
    var_k = "k"
    var_carry = "carry"
    var_carry2 = "carry2"
    var_m_prime = "m_prime"
    var_m = lambda i: f"m.c{i}"

    lines = []

    lines.append(f"u32 {var_k}, {var_carry};")
    lines.append(f"u32 {var_carry2} = 0;")

    for i in range(n_words):
        lines.append(f"{var_k} = {var_r(i)} * {var_m_prime};")
        lines.append(f"{var_carry} = 0;")
        lines.append(f"madc({var_r(i)}, {var_k}, {var_m(0)}, {var_carry});")
        for j in range(1, n_words):
            lines.append(f"{var_r(i + j)} = madc({var_r(i + j)}, {var_k}, {var_m(j)}, {var_carry});")
        lines.append(f"{var_r(i + n_words)} = addc({var_r(i + n_words)}, {var_carry2}, {var_carry});")
        lines.append(f"{var_carry2} = {var_carry};")

    return lines;


T = TypeVar("T")
def index_of_predicate(lst: List[T], predicate: Callable[[T], bool], begin: int = 0) -> Optional[int]:
    for i, item in enumerate(lst):
        if i >= begin and predicate(item):
            return i
    return None
    
def count_indentation(s: str) -> int:
    cnt = 0
    while s[cnt] == ' ':
        cnt += 1
    return cnt

def insert_generated(lines: List[str], point_name: str, to_insert: List[str]) -> List[str]:
    i = index_of_predicate(lines, lambda line: line.lstrip().rstrip() == f"// >>> GENERATED: {point_name}")
    if i is None:
        raise RuntimeError(f"Point {point_name} not found in document")
    j = index_of_predicate(lines, lambda line: line.lstrip().rstrip() == "// >>> GENERATED END", begin = i)
    if j is None:
        raise RuntimeError(f"Point {point_name} does not end")
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
    code_lines = insert_generated(
        code_lines,
        "square",
        square(8).to_lines()
    )
    code_lines = insert_generated(
        code_lines,
        "host_mul",
        host_mul(8)
    )
    code_lines = insert_generated(
        code_lines,
        "host_square",
        host_square(8)
    )
    code_lines = insert_generated(
        code_lines,
        "host_montgomery_reduction",
        host_montgomery_reduction(8)
    )

    with open(target, "w") as wf:
        wf.write("\n".join(code_lines))

