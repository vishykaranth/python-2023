from lib.files import read_csv_matrix_file

data = read_csv_matrix_file('../input/path_matrix.txt')
data_map = [[None for x in range(len(data))] for y in range(len(data))]

END_POINT = len(data) - 1


def in_bounds(row, col):
    return 0 <= row < len(data) and 0 <= col < len(data)


def solve(row, col):
    if row == END_POINT and col == END_POINT:
        return data[row][col]

    if data_map[row][col] is not None:
        return data_map[row][col]

    right_path = data[row][col] + solve(row, col + 1) if in_bounds(row, col + 1) else 5_000_000
    down_path = data[row][col] + solve(row + 1, col) if in_bounds(row + 1, col) else 5_000_000

    data_map[row][col] = min(right_path, down_path)
    return min(right_path, down_path)


print(solve(0, 0))
