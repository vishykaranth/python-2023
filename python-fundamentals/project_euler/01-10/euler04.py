from project_euler.lib.strings import is_palindrome


def largest_palindrome():
    largest = 0

    for x in range(999, 0, -1):
        for y in range(999, 0, -1):
            if x * y > largest and is_palindrome(x * y):
                largest = x * y

    return largest


print(largest_palindrome())
