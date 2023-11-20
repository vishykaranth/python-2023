from math import ceil


def is_right_angle(x, y, z):
    return x ** 2 + y ** 2 == z ** 2


def get_triplet(num):

    # Performance improved by considering the below rules
    # X would never be > 1/3 of num
    # Y would never be > 1/2 of num
    # Z would never be > 1/2 of num
    for x in range(1, int(ceil(num / 3))):
        for y in range(x + 1, int(ceil(num / 2))):
            for z in range(y + 1, int(ceil(num / 2))):
                if x + y + z == num and is_right_angle(x, y, z):
                    return x * y * z

    return False


print(get_triplet(1000))
