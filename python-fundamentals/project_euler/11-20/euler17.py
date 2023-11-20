
number_map = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
            17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
            60: "sixty", 70: "seventy", 80: "eighty",  90: "ninety"}


def count_letters(string):
    return len(str(string).replace(' ', '').replace('-', ''))


def convert_to_string(number):
    string = ""

    while number > 0:
        if number > 999:
            string += "one thousand"
            number -= number
        elif number >= 100:
            val = int(str(number)[0])
            string += number_map[val] + " hundred"
            number %= 100
        elif number >= 20:
            val = int(str(number)[0])*10
            string += number_map[val]
            number %= val
            if number > 0 and number < 10:
                string += "-" + number_map[number]
                number -= number
        elif number > 10:
            string += number_map[number]
            number %= number
        elif number <= 10:
            string += number_map[number]
            number -= number

        if number > 0:
            string += " and "

    return string

print(sum(count_letters(convert_to_string(x)) for x in range(1, 1000 + 1, 1)))