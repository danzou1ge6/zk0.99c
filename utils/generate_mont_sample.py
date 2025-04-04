from random import randint
import math
import argparse

from common import *

def m_magic(m: int, n_words: int) -> int:
    x, y, gcd = ext_gcd(m, 2 ** (n_words * 32))
    return (-x) % (2 ** 32)

def montgomery_reduction(x: int, bits: int, m: int) -> int:
    return (x * inv_modulo(2 ** bits, m)) % m

def params(bits: int, m: int) -> str:
    n_words = math.ceil(bits / 32)
    m_prime = m_magic(m, n_words=n_words)
    r = 2 ** bits
    r_mod = r % m
    r2_mod = r * r % m
    s = ''
    s += f"// {m}\n"
    s +=  "namespace device_constants {\n"
    s += f"  const __device__ Number m = BIG_INTEGER_CHUNKS8({chunks(m, n_words=n_words)});\n"
    s += f"  const __device__ Number mm2 = BIG_INTEGER_CHUNKS8({chunks(m * 2, n_words=n_words)});\n"
    s += f"  const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8({chunks(m - 2, n_words=n_words)});\n"
    s += f"  const __device__ Number r_mod = BIG_INTEGER_CHUNKS8({chunks(r_mod, n_words=n_words)});\n"
    s += f"  const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8({chunks(r2_mod, n_words=n_words)});\n"
    s +=  "};\n"

    s +=  "namespace host_constants {\n"
    s += f"  const Number m = BIG_INTEGER_CHUNKS8({chunks(m, n_words=n_words)});\n"
    s += f"  const Number mm2 = BIG_INTEGER_CHUNKS8({chunks(m * 2, n_words=n_words)});\n"
    s += f"  const Number m_sub2 = BIG_INTEGER_CHUNKS8({chunks(m - 2, n_words=n_words)});\n"
    s += f"  const Number r_mod = BIG_INTEGER_CHUNKS8({chunks(r_mod, n_words=n_words)});\n"
    s += f"  const Number r2_mod = BIG_INTEGER_CHUNKS8({chunks(r2_mod, n_words=n_words)});\n"
    s +=  "};\n"

    s += f"const u32 m_prime = {m_prime};\n"
    return s

def one_sample(bits: int, m: int) -> str:
    assert m < 2 ** bits
    r = 2 ** bits

    a = randint(0, m - 1)
    b = randint(0, m - 1)
    sub_ab = a - b if a >= b else a + 2 ** bits - b
    sum_ab = a + b
    sum_ab_mont = (a + b) % m
    sub_ab_mont = a - b if a >= b else a + m - b
    product = a * b
    a_square = a * a
    a_square_mont = (a * a * r) % m
    pow_mont = (quick_pow(a, m, b) * r) % m
    a_inv_mont = (quick_pow(a, m, m - 2) * r) %m
    product_mont = montgomery_reduction(product, bits, m)

    n_words = math.ceil(bits / 32)
    
    s = ""
    s += params(bits, m)
    s += f"// {a}\n"
    s += f"const u32 a[WORDS] = BIG_INTEGER_CHUNKS8({chunks(a, n_words=n_words)});\n"
    s += f"// {b}\n"
    s += f"const u32 b[WORDS] = BIG_INTEGER_CHUNKS8({chunks(b, n_words=n_words)});\n"
    s += f"const u32 sum[WORDS] = BIG_INTEGER_CHUNKS8({chunks(sum_ab, n_words=n_words)});\n"
    s += f"const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(sum_ab_mont, n_words=n_words)});\n"
    s += f"const u32 sub[WORDS] = BIG_INTEGER_CHUNKS8({chunks(sub_ab, n_words=n_words)});\n"
    s += f"const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(sub_ab_mont, n_words=n_words)});\n"
    s += f"const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS16({chunks(product, n_words=n_words * 2)});\n"
    s += f"const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(product_mont, n_words=n_words)});\n"
    s += f"const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS16({chunks(a_square, n_words=n_words * 2)});\n"
    s += f"const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(a_square_mont, n_words=n_words)});\n"
    s += f"const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(pow_mont, n_words=n_words)});\n"
    s += f"const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS8({chunks(a_inv_mont, n_words=n_words)});\n"
    s += f"const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS8({chunks(a >> 125, n_words=8)});\n"
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("-m", type=int, default=0)
    args = parser.parse_args()

    print(one_sample(args.bits, args.m))

