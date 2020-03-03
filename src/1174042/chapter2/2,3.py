# use one-hot encoding on categorical columns
muaraenim = pd.get_dummies(muaraenim, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])
muaraenim.head()
