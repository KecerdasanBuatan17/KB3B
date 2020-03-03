# shuffle rows
muaraenim = muaraenim.sample(frac=1)
# split training and testing data
muaraenim_train = muaraenim[:500]
muaraenim_test = muaraenim[500:]
muaraenim_train_att = muaraenim_train.drop(['pass'], axis=1)
muaraenim_train_pass = muaraenim_train['pass']
muaraenim_test_att = muaraenim_test.drop(['pass'], axis=1)
muaraenim_test_pass = muaraenim_test['pass']
muaraenim_att = muaraenim.drop(['pass'], axis=1)
muaraenim_pass = muaraenim['pass']
# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(muaraenim_pass), len(muaraenim_pass),100*float(np.sum(muaraenim_pass)) / len(muaraenim_pass)))
