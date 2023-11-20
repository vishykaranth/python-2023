
def palidrome(num):
    return str(num)[::-1] == str(num)


def is_lychrel(num, iterations):
    if iterations > 50:
        return False

    new_num = int(str(num)[::-1]) + num
    if palidrome(new_num):
        return True

    return is_lychrel(new_num, iterations + 1)


print(10000 - len([x for x in range(1, 10001) if is_lychrel(x, 1)]))

