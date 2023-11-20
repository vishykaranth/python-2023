def irrational_string(length):
    string = ""
    counter = 1

    while len(string) < length:
        string += str(counter)
        counter += 1

    return string[length - 1]


print(
    int(irrational_string(1)) *
    int(irrational_string(10)) *
    int(irrational_string(100)) *
    int(irrational_string(1000)) *
    int(irrational_string(10000)) *
    int(irrational_string(100000)) *
    int(irrational_string(1000000))
)
