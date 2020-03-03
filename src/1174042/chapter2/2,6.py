# visualize tree
import graphviz
dot_data = tree.export_graphviz(palembang, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(muaraenim_train_att), class_names=["fail", "pass"],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph