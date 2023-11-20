
largest = 0

for x in range(1, 100):
    for y in range(1, 100):
        num = x ** y
        digit_sum = sum([int(n) for n in str(num)])
        if digit_sum > largest:
            largest = digit_sum

print(largest)