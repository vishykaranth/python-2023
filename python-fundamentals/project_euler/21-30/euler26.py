import decimal

best_length = 0
best_num = 0


def length_of_cycle(num):
    first = 1 % num
    last = -1
    count = 0

    while first is not last and count < num - 1:
        if last is -1:
            last = first

        last = (last * 10) % num
        count += 1

    return count

print(length_of_cycle(5))
#
# for x in range(1, 1000):
#     length = length_of_cycle(x)
#     if length > best_length:
#         best_length = length
#         best_num = x
#
# print("{} is the has the longest sequence of {} characters!".format(best_num, best_length))
