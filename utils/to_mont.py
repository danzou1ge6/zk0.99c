from common import *

def to_mont(x: int, m: int, r: int) -> int:
    return (x * r ) % m


n_words = 12
bits = 384
# m = 21888242871839275222246405745257275088696311157297823662689037894645226208583
m = 0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001

a = to_mont(3, m, 2**bits)
print(f"BIG_INTEGER_CHUNKS8({chunks(a, n_words=n_words)})")