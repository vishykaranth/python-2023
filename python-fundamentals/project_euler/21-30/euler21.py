divisor_map = {}


def divisors(n):
    divs = divisor_map.get(n, False)
    if divs is not False:
        return divs

    divisor_map[n] = sum([x for x in range(1, n, 1) if n % x == 0])
    return divisor_map[n]


def amicable_nums(max_range):
    nums = set()

    for x in range(1, max_range, 1):
        for y in range(1, max_range, 1):
            if x == y:
                continue
            x_divs = divisors(x)
            y_divs = divisors(y)

            if x_divs == y and y_divs == x:
                nums.add(x)
                nums.add(y)

    return nums

print(sum(amicable_nums(10000)))