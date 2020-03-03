# generate binary label (pass/fail) based on G1+G2+G3
# (test grades, each 0-20 pts); threshold for passing is sum>=30
muaraenim['pass'] = muaraenim.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3'])
>= 35 else 0, axis=1)
muaraenim = muaraenim.drop(['G1', 'G2', 'G3'], axis=1)
print(muaraenim.head())