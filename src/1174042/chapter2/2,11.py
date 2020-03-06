depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    palembang = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(palembang, muaraenim_att, muaraenim_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2
    i += 1

print(depth_acc)