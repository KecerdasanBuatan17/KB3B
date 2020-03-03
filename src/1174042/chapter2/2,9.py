from sklearn.model_selection import cross_val_score
scores = cross_val_score(palembang, muaraenim_att, muaraenim_pass, cv=5)
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))