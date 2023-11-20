from project_euler.lib.fact import fact_cached


def find_digit_factorials():
    total = 0

    for x in range(3, 3000000):
        digits = list(str(x))
        digit_sum = sum([fact_cached(int(x)) for x in digits])
        if digit_sum == x:
            total += digit_sum

    return total

print(find_digit_factorials())