# save tree
tree.export_graphviz(palembang, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(muaraenim_train_att), class_names=["fail", "pass"], filled=True, rounded=True)
