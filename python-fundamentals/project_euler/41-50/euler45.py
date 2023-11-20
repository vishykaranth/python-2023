
def gen_triangles(max_length):
    return [int(x * (x + 1) / 2) for x in range(1, max_length + 1)]


def gen_pentagonals(max_length):
    return [int(x * (x * 3 - 1) / 2) for x in range(1, max_length + 1)]


def gen_hexagonals(max_length):
    return [int(x * (x * 2 - 1)) for x in range(1, max_length + 1)]


def find_next():
    for t in triangles:
        # Need to find next number after the provided example of 40755
        if t <= 40755:
            continue
        if t in pentagonals and t in hexagonals:
            return t

    return None

triangles = gen_triangles(100000)
pentagonals = gen_pentagonals(100000)
hexagonals = gen_hexagonals(1000000)

print(find_next())