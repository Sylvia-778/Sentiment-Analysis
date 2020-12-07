import csv


line1 = []
line2 = []
output = open('output.txt', 'r')
for i in output:
    line1.append(i.split()[1])
test = open('test.tsv', 'r')
csv_reader = csv.reader(test, delimiter='\t')
for i in csv_reader:
    line2.append(i[2])


count = 0
for i in range(len(line2)):
    if line1[i] != line2 [i]:
        print(4001+i, line1[i], line2[i])
        count += 1
print(count)
