from generate_ec_point import *
from random import randint
from sys import argv

if __name__ == "__main__":
    msm_len = 2 ** int(argv[1])

    wf = open(f'msm/tests/msm{argv[1]}.input', 'w')

    wf.write(f"{msm_len}\n")

    for i in range(msm_len):
        s = randint(0, m - 1)
        x, y = random_point()
        x_mont = (x * r) % m
        y_mont = (y * r) % m
        wf.write(f'{chunked_hex(s, 8)}|{chunked_hex(x_mont, 8)} {chunked_hex(y_mont, 8)}\n')
    
    wf.close()
        
