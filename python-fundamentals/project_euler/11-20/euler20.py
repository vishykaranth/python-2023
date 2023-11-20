from project_euler.lib.fact import fact

digits = fact(100)
print(sum(int(x) for x in list(str(digits))))