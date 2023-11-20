# Todo: logic to solve this problem is correct, but takes an EXTREMELY long time to run.


def odd_digits(num):
    if num % 2 == 0:
        return False

    for c in str(num):
        if int(c) % 2 == 0:
            return False

    return True


def reverseable(num):
    # Leading zeros not allowed so the number cannot be divisible by 10
    if num % 10 == 0:
        return

    # Already cached, no need to check again
    if num in reversable_map:
        return

    new_num = int(str(num)[::-1]) + num
    if odd_digits(new_num):
        reversable_map.add(num)
        reversable_map.add(int(str(num)[::-1]))


reversable_map = set()

for x in range(1, 1000000000):
    reverseable(x)

print(len(reversable_map))


