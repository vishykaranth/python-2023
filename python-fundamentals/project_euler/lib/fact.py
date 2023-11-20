
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)


fact_map = {}


def fact_cached(n):
    if n <= 1:
        return 1
    if n in fact_map:
        return fact_map[n]

    fact_map[n] = n * fact_cached(n - 1)
    return fact_map[n]
