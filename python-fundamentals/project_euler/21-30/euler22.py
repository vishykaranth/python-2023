from project_euler.lib.files import read_single_line_file
from project_euler.lib.strings import str_ordinal_value


def calculate_score(name, index):
    return str_ordinal_value(name) * (index + 1)

names = sorted(read_single_line_file("../input/names.txt"))
print(sum(calculate_score(name, index) for index, name in enumerate(names)))
