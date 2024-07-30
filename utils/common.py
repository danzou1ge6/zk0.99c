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

def quick_pow(x: int, m: int, p: int) -> int:
    r = 1
    while p != 0:
        if p & 1 == 1:
            r = (r * x) % m
        x = (x * x) % m
        p = p >> 1
    return r
