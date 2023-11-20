from math import *


def is_prime(x):
    if x == 2:
        return True
    if x < 2 or x % 2 == 0:
        return False

    for y in range(3, ceil(sqrt(x)) + 1, 2):
        if x % y == 0:
            return False

    return True

print(2 + sum(x for x in range(3, 2000000, 2) if is_prime(x)))