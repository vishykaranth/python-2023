from project_euler.lib.primes import is_prime


def convert_to_rotations(num):
    num = str(num)
    return set(num[x:] + num[:x] for x in range(len(num)))


def find_circular_primes():
    # Start a 1 due to the edge case of 2 being prime
    total = 1

    for x in range(3, 1000000, 2):
        if is_prime(x):
            rotations = convert_to_rotations(x)
            total += 1 if len([y for y in rotations if is_prime(int(y))]) == len(rotations) else 0

    return total

print(find_circular_primes())
