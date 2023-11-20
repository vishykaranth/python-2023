from lib.files import read_multiline_file

# Sudoku board
board = [[0 for x in range(0, 9)] for y in range(0, 9)]

# Read the raw input file
data = read_multiline_file("../input/sudoku.txt")
# Remove the grid headers from the input
data[:] = (x for x in data if x.count('Grid') == 0)


def print_board():
    for x in range(0, len(board)):
        for y in range(0, len(board)):
            print("{}".format(board[x][y]), end='')
        print("")
    print("")


def validate(row, col, num):
    # Check in full row or column
    for x in range(0, len(board)):
        if board[row][x] == num or board[x][col] == num:
            return False

    # Check in 3x3 grid
    r = int(row / 3) * 3
    c = int(col / 3) * 3
    for x in range(r, r + 3):
        for y in range(c, c + 3):
            if board[x][y] == num:
                return False

    return True


def puzzle_solved():
    for x in range(0, len(board)):
        for y in range(0, len(board)):
            if board[x][y] == 0:
                return False

    return True


def solve_next(row, col):
    if col >= 8:
        return solve(row + 1, 0)
    else:
        return solve(row, col + 1)


def solve(row, col):
    if puzzle_solved():
        return True
    elif board[row][col] > 0:
        return solve_next(row, col)
    else:
        for x in range(1, 10):
            if validate(row, col, x):
                board[row][col] = x
                if solve_next(row, col):
                    return True
                board[row][col] = 0
    return False


def process_puzzles():
    digit_sum = 0

    # Process all 50 puzzles
    for p in range(0, 50):
        # Build the sudoku board
        for x in range(0, 9):
            for y in range(0, 9):
                board[x][y] = int(data[(p * 9) + x][y])

        # Solve the puzzle
        print_board()
        solve(0, 0)
        print_board()

        # Sum the digits in the top left as per the question
        digit_sum += int(''.join(str(board[0][0]) + str(board[0][1]) + str(board[0][2])))

    return digit_sum


print(process_puzzles())