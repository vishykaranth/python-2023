from lib.primes import is_prime
from itertools import permutations


def find_pandigital_prime():
    largest = 0
    for x in range(1, 10):
        string = ''.join([str(y) for y in range(1, x + 1)])
        perms = [int(''.join(z)) for z in permutations(string)]

        for p in perms:
            if p > largest and is_prime(p):
                largest = p

    return largest

print(find_pandigital_prime())
