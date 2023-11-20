
def contains_same_digits(x, y):
    x = str(x)
    y = str(y)

    if len(x) != len(y):
        return False

    return len([d for d in x if x.count(d) != y.count(d)]) == 0


def find_smallest():
    num = 1

    while True:
        if len([x for x in range(2, 7) if not contains_same_digits(num, x * num)]) == 0:
            return num
        num += 1


print(find_smallest())
