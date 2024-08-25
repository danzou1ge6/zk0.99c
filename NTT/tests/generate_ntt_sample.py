# BLS12-381
#[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[PrimeFieldGenerator = "7"]

from sympy import ntt 
import random
import argparse

def printChunk(num):
    for j in range(8):
        chunk = num >> (32 * j) & 0xFFFFFFFF
        print(chunk, end = " ")
    print("")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate NTT sample')
    parser.add_argument('--log_len', type=int, default=2, help='Length of the sequence')

    prime = 52435875175126190479447740508185965837690552500527637822603658699938581184513

    random.seed(0)

    len = 2**parser.parse_args().log_len

    seq = []

    for i in range(len):
        num = 0
        for j in range(8):
            tmp = random.randint(0, 2**32)
            num = (num << 32) + tmp
        num = num % prime
        seq.append(num)

    for i in range(len):
        printChunk(seq[i])
  
    # ntt 
    transform = ntt(seq, prime) 

    for i in range(len):
        printChunk(transform[i])