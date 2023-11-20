from lib.numbers import is_bouncy


bouncy_count = 0
x = 1
while True:
    if is_bouncy(x):
        bouncy_count += 1

    if x > 0 and (bouncy_count * 1.0 / x) == 0.99:
        print(x)
        break

    x += 1

