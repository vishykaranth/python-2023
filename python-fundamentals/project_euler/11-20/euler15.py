
GRID_SIZE = 21
grid = [[x for x in range(0, GRID_SIZE)] for y in range(0, GRID_SIZE)]
grid_map = [[None for x in range(0, GRID_SIZE)] for y in range(0, GRID_SIZE)]


def lattice_path(x, y):
    if not in_bounds(x, y):
        return 0

    if x == (GRID_SIZE - 1) and y == (GRID_SIZE - 1):
        return 1

    if grid_map[x][y] is not None:
        return grid_map[x][y]

    grid_map[x][y] = lattice_path(x + 1, y) + lattice_path(x, y + 1)
    return grid_map[x][y]
    

def in_bounds(x, y):
    if x < 0 or x >= GRID_SIZE:
        return False
    if y < 0 or y >= GRID_SIZE:
        return False
    return True


print(lattice_path(0, 0))
