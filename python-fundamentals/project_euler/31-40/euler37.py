from lib.primes import is_prime


def check_num(num):
    num = str(num)

    for x in range(0, len(num)):
        left = num[x:len(num)]
        right = num[0:x+1]
        if not is_prime(int(left)) or not is_prime(int(right)):
            return False

    return True


def find_truncatable_primes():
    primes = []
    # Primes below 7 aren't included so start from the next odd number
    step = 9

    while len(primes) < 11:
        if is_prime(step) and check_num(step):
            primes.append(step)
        step += 2

    return sum(primes)


print(find_truncatable_primes())