from lib.files import read_multiline_file


highest_val = 0
line_num = 0
highest_line_num = 0
lines = read_multiline_file('../input/exponents.txt')

for l in lines:
    d = l.split(',')
    x = int(d[0])
    y = int(d[1])
    z = pow(x, y, 1000)
    print('{} x={}, y={}, z={}'.format(line_num, x, y, z))
    if z > highest_val:
        highest_val = z
        highest_line_num = line_num

    line_num += 1

print(highest_val)
print(highest_line_num)