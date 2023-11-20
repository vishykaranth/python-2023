from project_euler.lib.strings import is_pandigital


def find_pandigital_products():
    pans = set()

    for x in range(1, 2500):
        for y in range(1, 2500):
            string = str(x) + str(y) + str(x * y)

            if len(string) is not 9:
                continue

            if is_pandigital(string, 9):
                pans.add(x*y)

    return sum(pans)

print(find_pandigital_products())
