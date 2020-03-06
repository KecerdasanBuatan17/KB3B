# fit a decision tree
from sklearn import tree
palembang = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
palembang = palembang.fit(muaraeni,_train_att, muaraeni_train_pass)
