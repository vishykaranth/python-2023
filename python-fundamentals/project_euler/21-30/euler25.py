
def fib(length):
    fib_num = 1
    prev_fib_num = 1
    index = 2

    while len(str(fib_num)) < length:
        temp = fib_num
        fib_num += prev_fib_num
        prev_fib_num = temp
        index += 1

    return index

print(fib(1000))