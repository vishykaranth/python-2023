def fib(upper):
    seq = []
    x = 1
    y = 1

    while y < upper:
        temp = x
        x = y
        y += temp
        seq.append(x)

    return seq


fibs = fib(4000000)
print(sum(x for x in fibs if x % 2 == 0))
