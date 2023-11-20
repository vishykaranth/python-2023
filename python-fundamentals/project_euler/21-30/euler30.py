

def check_power(number):
    return sum(int(x) ** 5 for x in list(str(number))) == number

print(sum(x for x in range(2, 999999, 1) if check_power(x)))
