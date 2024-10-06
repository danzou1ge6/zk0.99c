from typing import Tuple, Union
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
    z: int

@dataclass
class PointAffine:
    x: int
    y: int

@dataclass
class Curve:
    m: int
    a: int
    b: int

BN254_G1 = Curve(21888242871839275222246405745257275088696311157297823662689037894645226208583, 0, 3)

def padd(a: PointAffine, b: PointAffine, curve: Curve) -> PointAffine:
    m = curve.m
    if a.x == b.x and a.y == b.y:
        lam = (3 * a.x * a.x + curve.a) % m
        lam = (lam * inv_modulo(2 * a.y, curve.m)) % m
    else:
        lam = (b.y - a.y) % m
        lam = (lam * inv_modulo(b.x - a.x, curve.m)) % m
    x3 = (lam * lam - a.x - b.x) % m
    y3 = (lam * (a.x - x3) - a.y) % m
    return PointAffine(x3, y3)

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

def pad_zero_left(s: str, total_len: int):
    n_zeros = total_len - len(s)
    return '0' * n_zeros + s

def chunked_hex(x: int, n_words: int) -> str:
    s = hex(x)[2:]
    c = []
    while len(s) > 8:
        n_words  -= 1
        c.insert(0, s[-8:])
        s = s[:-8]
    n_words -= 1
    c.insert(0, pad_zero_left(s, 8))
    for _ in range(n_words):
        c.insert(0, "00000000")
    return '0x' + '_'.join(c)

def ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
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
