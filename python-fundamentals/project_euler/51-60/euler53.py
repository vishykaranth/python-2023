from lib.fact import *


def combinatoric(n, r):
    return int(fact_cached(n) / (fact_cached(r) * fact_cached(n - r)))

over_one_million = []

for x in range(1, 101):
    for y in range(1, x + 1):
        c = combinatoric(x, y)
        if c > 1000000:
            over_one_million.append(c)

print(len(over_one_million))
