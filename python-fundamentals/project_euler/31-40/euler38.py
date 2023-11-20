from lib.strings import is_pandigital


largest_pandigital = 0
largest_str = ''
num = 9
done = False

while not done:
    current = ''

    for x in range(1, num):
        current += str(num * x)

        # Gone over 9 characters so cannot be a 9 digit pandigital
        if len(current) > 9:
            break

        # Gone through enough numbers now, should be done
        if int(current) >= 987654321:
            done = True

        if is_pandigital(current, 9) and int(current) > largest_pandigital:
            largest_pandigital = num
            largest_str = current

    num += 1

print(largest_pandigital)
print(largest_str)


