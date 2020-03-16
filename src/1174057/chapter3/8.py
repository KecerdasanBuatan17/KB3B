from sklearn.model_selection import cross_val_score
scores = cross_val_score(bdg, dago_att, dago_pass, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))