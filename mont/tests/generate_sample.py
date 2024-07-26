from random import randint
import math

def chunks(x: int, n_words: int) -> str:
    s = hex(x)[2:]
    c = []
    while len(s) > 8:
        n_words  -= 1
        c.insert(0, '0x' + s[-8:])
        s = s[:-8]
    n_words -= 1
    c.insert(0, '0x' + s)
    for _ in range(n_words):
        c.insert(0, "0x0")
    return ', '.join(c)

def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:          
        return 1, 0, a     
    else:         
        x, y, gcd = ext_gcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, gcd

def inv_modulo(k: int, m: int) -> int:
    x, y, gcd = ext_gcd(k, m)
    assert gcd == 1
    return x

def m_magic(m: int, n_words: int) -> int:
    x, y, gcd = ext_gcd(m, 2 ** (n_words * 32))
    return (-x) % (2 ** 32)

def montgomery_reduction(x: int, bits: int, m: int) -> int:
    return (x * inv_modulo(2 ** bits, m)) % m


def one_sample(bits: int, m: int) -> str:
    assert m < 2 ** bits
    a = randint(0, m - 1)
    b = randint(0, m - 1)
    sub_ab = a - b if a >= b else a + 2 ** bits - b
    sum_ab = a + b
    sum_ab_mont = (a + b) % m
    sub_ab_mont = sub_ab % m
    product = a * b

    n_words = math.ceil(bits / 32)
    m_prime = m_magic(m, n_words=n_words)
    product_mont = montgomery_reduction(product, bits, m)

    r = 2 ** bits
    r_mod = r % m
    r2_mod = r * r % m
    
    s = ""
    s += f"// {m}\n"
    s +=  "const auto params = Params {\n"
    s += f"  .m = BIG_INTEGER_CHUNKS({chunks(m, n_words=n_words)}),\n"
    s += f"  .r_mod = BIG_INTEGER_CHUNKS({chunks(r_mod, n_words=n_words)}),\n"
    s += f"  .r2_mod = BIG_INTEGER_CHUNKS({chunks(r2_mod, n_words=n_words)}),\n"
    s += f"  .m_prime = {m_prime}\n"
    s +=  "};\n"
    s += f"// {a}\n"
    s += f"const u32 a[WORDS] = BIG_INTEGER_CHUNKS({chunks(a, n_words=n_words)});\n"
    s += f"// {b}\n"
    s += f"const u32 b[WORDS] = BIG_INTEGER_CHUNKS({chunks(b, n_words=n_words)});\n"
    s += f"const u32 sum[WORDS] = BIG_INTEGER_CHUNKS({chunks(sum_ab, n_words=n_words)});\n"
    s += f"const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS({chunks(sum_ab_mont, n_words=n_words)});\n"
    s += f"const u32 sub[WORDS] = BIG_INTEGER_CHUNKS({chunks(sub_ab, n_words=n_words)});\n"
    s += f"const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS({chunks(sub_ab_mont, n_words=n_words)});\n"
    s += f"const u32 prod[WORDS] = BIG_INTEGER_CHUNKS({chunks(product, n_words=n_words)});\n"
    s += f"const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS({chunks(product_mont, n_words=n_words)});\n"
    return s

if __name__ == "__main__":
    p = 8749054268878081845992735117657085490803352298049958388812839746200388362933
    print(one_sample(256, p))

