from math import ceil, sqrt


def is_prime(x):
    if x == 2:
        return True
    if x < 2 or x % 2 == 0:
        return False

    for y in range(3, ceil(sqrt(x)) + 1, 2):
        if x % y == 0:
            return False

    return True


prime_map = {}


def is_prime_cached(x):
    if x == 2:
        return True
    if x < 2 or x % 2 == 0:
        return False

    if x in prime_map:
        return prime_map[x]

    for y in range(3, ceil(sqrt(x)) + 1, 2):
        if x % y == 0:
            prime_map[x] = False
            return False

    prime_map[x] = True
    return True


def gen_primes(max_range):
    primes = [2]

    for x in range(3, max_range, 2):
        if is_prime(x):
            primes.append(x)

    return primes
