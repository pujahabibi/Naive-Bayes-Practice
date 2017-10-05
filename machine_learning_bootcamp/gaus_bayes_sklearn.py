import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


# Create an empty dataframe
data = pd.DataFrame()

# Create our target variable
data['Gender'] = ['male','male','male','male','female','female','female','female']

# Create our feature variables
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]

# View the data
print(data)
print(data.describe())
X = np.array([[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10],
              [5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]])
Y = np.array(['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'])

clf = GaussianNB()
print(clf.fit(X, Y))
print(clf.predict([[6, 130, 8]]))