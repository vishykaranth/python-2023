from itertools import permutations


def char_ordinal_value(char):
    return ord(char) - 65 + 1


def str_ordinal_value(string):
    return sum(ord(x) - 65 + 1 for x in list(string.upper()))


def is_palindrome(num):
    # fancy way to reverse a string
    return str(num) == str(num)[::-1]


def is_pandigital(number, n):
    number = str(number)

    if len(number) is not n:
        return False

    for x in range(1, n + 1):
        if number.count(str(x)) is not 1:
            return False

    return True


def get_permutations(number, as_int=False):
    if as_int:
        return [int(''.join(x)) for x in permutations(str(number))]
    return [''.join(x) for x in permutations(str(number))]