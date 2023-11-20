from lib.strings import is_palindrome


def to_binary(num):
    return str(bin(num))[2:]


print(sum(x for x in range(1, 1000000) if is_palindrome(x) and is_palindrome(to_binary(x))))