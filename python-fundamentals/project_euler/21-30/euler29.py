
def power_combinations(min_range, max_range):
    combos = set()

    for x in range(min_range, max_range+1, 1):
        for y in range(min_range, max_range+1, 1):
            combos.add(x ** y)
            combos.add(y ** x)

    return sorted(combos)

print(len(power_combinations(2, 100)))