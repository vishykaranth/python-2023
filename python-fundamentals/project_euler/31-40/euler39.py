from math import ceil


def is_right_angle(x, y, z):
    return x ** 2 + y ** 2 == z ** 2


def get_permutations():
    perm_map = {}

    for p in range(0, 1000):
        perms = []

        # Performance improved by considering the below rules
        # however the performance could still be greatly improved.
        # X would never be > 1/3 of p
        # Y would never be > 1/2 of p
        # Z would never be > 1/2 of p
        for x in range(1, int(ceil(p / 3))):
            for y in range(x + 1, int(ceil(p / 2))):
                for z in range(y + 1, int(ceil(p / 2))):
                    if x + y + z == p and is_right_angle(x, y, z):
                        perms.append((x, y, z))

        perm_map[p] = len(perms)
        print("Finished checking {}".format(p))

    print(perm_map)
    return max(perm_map, key=perm_map.get)


print(get_permutations())
