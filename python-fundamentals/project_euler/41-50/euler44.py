def gen_pentagonals(max_length):
    return [int(x * (x * 3 - 1) / 2) for x in range(1, max_length + 1)]


def find_pentagon():
    # Begin with some arbitrarily large number, 1 billion in this case
    min_value = 1000000000

    for x in pentagonals:
        for y in pentagonals:
            if x == y:
                continue

            if x + y in pentagonals:
                if abs(y - x) in pentagonals:
                    if abs(y - x) < min_value:
                        min_value = abs(y - x)

    return min_value

pentagonals = gen_pentagonals(2500)
print(find_pentagon())