from math import *


def sum_of_squares(min, max):
    return sum(x**2 for x in range(min, max+1))


def square_of_sum(min, max):
    return int(pow(sum(x for x in range(min, max+1)), 2))

print(square_of_sum(1, 100) - sum_of_squares(1, 100))
