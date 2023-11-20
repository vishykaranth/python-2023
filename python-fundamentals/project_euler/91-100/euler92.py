
# Prefill the cache with the examples we are given
chain_map = {44: 1, 85: 89}


def chain(num):
    if num is 1 or num is 89:
        return num

    if num not in chain_map:
        chain_map[num] = chain(sum(int(d) ** 2 for d in str(num)))

    return chain_map[num]


print(len([n for n in range(1, 10000000) if chain(n) == 89]))