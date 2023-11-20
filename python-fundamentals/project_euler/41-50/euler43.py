from itertools import permutations

perms = [''.join(x) for x in permutations("1406357289")]

total = 0

for p in perms:
    if int(p[1:4]) % 2 == 0:
        if int(p[2:5]) % 3 == 0:
            if int(p[3:6]) % 5 == 0:
                if int(p[4:7]) % 7 == 0:
                    if int(p[5:8]) % 11 == 0:
                        if int(p[6:9]) % 13 == 0:
                            if int(p[7:10]) % 17 == 0:
                                total += int(p)

print(total)
