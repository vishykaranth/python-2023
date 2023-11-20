def reverseString(str):
    str_list = list(str)
    start = 0
    len1 = len(str_list)
    end = len1 - 1
    mid = len1 // 2

    # for _ in range(mid + 1):
    for _ in range(mid):
        str_list[start], str_list[end] = str_list[end], str_list[start]
        start += 1
        end -= 1

    return "".join(str_list)


# print(reverseString("abc"))
# print(reverseString("abcddd"))
print(reverseString("ab"))

# https://docs.python.org/3.7/library/stdtypes.html#range