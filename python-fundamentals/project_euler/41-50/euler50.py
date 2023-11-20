from math import *

# Logic is correct but needs optimizing to produce a solution
# in a reasonable time.


def is_prime(x):
    if x == 2:
        return True

    if x < 2 or x % 2 == 0:
        return False

    for y in range(3, int(ceil(sqrt(x)) + 1), 2):
        if x % y == 0:
            return False

    return True


def gen_primes(max_range=100000):
    return [2] + [x for x in range(3, max_range, 2) if is_prime(x)]


def longest_consecutive_prime():
    prime_to_length = {}

    for p in primes:
        for bkey, b in enumerate(primes):
            total = 0
            length = 0

            for ckey, c in enumerate(primes):
                if ckey < bkey:
                    continue

                if total + c > p:
                    break

                total += c
                length += 1

            if total == p:
                print("{} has a length of {}".format(p, length))
                prime_to_length[p] = length
                break

    # print(prime_to_length)
    print(max(prime_to_length, key=prime_to_length.get))


primes = gen_primes(1000000)
print(longest_consecutive_prime())
