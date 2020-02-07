import pandas as pd
import sklearn
from sklearn import svm, preprocessing

df = pd.read_csv("datasets/diamonds.csv", index_col=0)

# Shuffle dataframe
df_sf = sklearn.utils.shuffle(df)

# Turn categorical dataframe's "features" into numerical for training (all values must be numerical)
# create dictionaries and then perform mapping
cut_dictionary = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_dictionary = {"J": 1, "I": 2, "H":  3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_dictionary = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL":11}

df_sf['cut'] = df_sf['cut'].map(cut_dictionary)
df_sf['color'] = df_sf['color'].map(color_dictionary)
df_sf['clarity'] = df_sf['clarity'].map(clarity_dictionary)

# Supervised Learning (we try to predict diamonds' prices)
X = df_sf.drop(columns = ['price']).values
y = df_sf['price'].values

# Perform "scaling" on data
X = preprocessing.scale(X)

# Split data into training and testing
test_size = 200

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

# Build and train our regression model 
# * Model chosen, based on https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html *
clf = svm.SVR(kernel = 'rbf')
clf.fit(X_train, y_train)

# Evaluation
clf.score(X_test, y_test)