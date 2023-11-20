from lib.primes import *


def is_permutation(x, y, z):
    x = str(x)
    y = str(y)
    z = str(z)

    # Make sure the values are different
    if x == y or x == z or y == z:
        return False

    # Make sure lengths match
    if len(x) is not len(y) or len(x) is not len(z):
        return False

    for index, char in enumerate(x):
        # Character doesn't appear in one of the strings
        if y.count(char) is not z.count(char) or z.count(char) is not x.count(char):
            return False

    return True


def find_prime_permutation():
    # 1009: first prime > 1000
    primes = [x for x in range(1009, 10000, 2) if is_prime(x)]

    for prime in primes:
        # Skip this permutation as the question wants us to find the other
        if prime == 1487:
            continue

        for y in range(1, 10000):
            a = prime + y
            b = prime + (y * 2)
            if is_prime(a) and is_prime(b) and is_permutation(prime, a, b):
                return str(prime) + str(a) + str(b)

    return None

print(find_prime_permutation())
