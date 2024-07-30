from typing import Optional, Tuple
from random import randint
from pathlib import Path

from common import *

m = 21888242871839275222246405745257275088696311157297823662689037894645226208583
k = 5472060717959818805561601436314318772174077789324455915672259473661306552146
a = 0
b = 3
r = 2 ** 256

def try_sqrt(x: int) -> Optional[int]:
  x_sqrt = quick_pow(x, m, k)
  if (x_sqrt * x_sqrt) % m == x:
    return x_sqrt
  return None

def random_point() -> Tuple[int, int]:
  while True:
    x = randint(0, m - 1)
    y2 = (x ** 3) % m + a * x + b
    y2 = y2 % m
    y = try_sqrt(y2)
    
    if y != None:
        if randint(0, 1) == 0:
            return x, y
        else:
            return x, (-y) % m

if __name__ == "__main__":
    x, y = random_point()
    # Convert to Montgomery representation
    x_mont = (x * r) % m
    y_mont = (y * r) % m
    print(f"// x = {x}")
    print(f"// x_mont = {x_mont}")
    print(f"const u32 x[8] = BIG_INTEGER_CHUNKS8({chunks(x_mont, 8)});")
    print(f"// y = {y}")
    print(f"// y_mont = {y_mont}")
    print(f"const u32 y[8] = BIG_INTEGER_CHUNKS8({chunks(y_mont, 8)});")

