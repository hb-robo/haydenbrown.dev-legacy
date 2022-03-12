import csv
reader = csv.DictReader(open('gen1types.csv'))

type_matchups = {}
for row in reader:
    key = row.pop('off_type')
    type_matchups[key] = row

print(type_matchups)