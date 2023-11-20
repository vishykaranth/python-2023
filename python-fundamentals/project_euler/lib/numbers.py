from math import sqrt


def is_bouncy(n):
    last_num = False
    increasing = True
    decreasing = True

    for c in str(n):
        int_c = int(c)

        if last_num is False:
            last_num = int_c
            continue

        if int_c < last_num:
            increasing = False

        if int_c > last_num:
            decreasing = False

        if not increasing and not decreasing:
            return True

        last_num = int_c

    return False


def is_composite(num):
    if num < 3:
        return False

    for x in range(2, int(sqrt(num)) + 1):
        if num % x == 0:
            return True

    return False


def gen_composite(max_range):
    comp = []

    # No composites below 4
    for x in range(4, max_range + 1):
        if is_composite(x):
            comp.append(x)

    return comp

