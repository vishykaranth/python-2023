from lib.files import read_single_line_file
from lib.strings import str_ordinal_value


def is_triangle_word(word):
    return str_ordinal_value(word) in triangles


def gen_triangle_numbers(max_range):
    return [int(0.5 * x * (x + 1)) for x in range(1, max_range + 1)]


triangles = gen_triangle_numbers(10000)
words = read_single_line_file("../input/words.txt")

print(len([x for x in words if is_triangle_word(x)]))