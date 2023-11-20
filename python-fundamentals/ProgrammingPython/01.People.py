bob = ['Bob Smith', 42, 30000, 'software']
sue = ['Sue Jones', 45, 40000, 'hardware']

# fetch name, pay
print(bob[0], sue[2])

# what's bob's last name?
print(bob[0].split()[-1])

# give sue a 25% raise
sue[2] *= 1.25
print(sue)

# reference in list of lists
people = [bob, sue]
for person in people:
    print(person)

print(people[1][0])

for person in people:
    print(person[0].split()[-1])  # print last names
    person[2] *= 1.20  # give each a 20% raise

print(people)

for person in people: print(person[2]) # check new pay