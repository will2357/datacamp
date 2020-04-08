import os

# Course deps
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# Open shared user defined functions
udf = "ml_techniques/01_supervised_scikit/udfs.py"
exec(open(os.path.abspath(udf)).read())

################################################################################
################### Ex. 2: Exploring categorical features ######################
################################################################################

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv("ml_techniques/01_supervised_scikit/_datasets/gapminder_2.csv")

# Create a boxplot of life expectancy per region
df.boxplot('life','Region', rot=60)

# Show the plot
plt.show()


################################################################################
####################### Ex. 3: Creating dummy variables ########################
################################################################################

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)


################################################################################
############### Ex. 4: Regression with categorical features ####################
################################################################################

X = df_region.drop('life', axis=1).values
y = df['life'].values

# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)


################################################################################
####################### Ex. 6: Dropping missing data ###########################
################################################################################
column_names = ['party',
                'infants',
                'water',
                'budget',
                'physician',
                'salvador',
                'religious',
                'satellite',
                'aid',
                'missile',
                'immigration',
                'synfuels',
                'education',
                'superfund',
                'crime',
                'duty_free_exports',
                'eaa_rsa']

df = pd.read_csv("ml_techniques/01_supervised_scikit/_datasets/house-votes-84.data", names=column_names)
df.replace(to_replace='y', value=1, inplace=True)
df.replace(to_replace='n', value=0, inplace=True)
# Did the following in chapter 1:
# df.replace(to_replace='?', value=np.nan, inplace=True)
# df.fillna(df.mean().round(0).astype(int), inplace=True)
# df[column_names[1:]] = df[column_names[1:]].astype(int)

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


################################################################################
############# Ex. 7: Imputing missing data in a ML Pipeline I ##################
################################################################################

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
         ('SVM', clf)]

################################################################################
############ Ex. 8: Imputing missing data in a ML Pipeline II ##################
################################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X = df.drop('party', axis=1)
y = df['party']

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
         ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


################################################################################
################# Ex. 10: Centering and scaling your data ######################
################################################################################

df = pd.read_csv("ml_techniques/01_supervised_scikit/_datasets/white-wine.csv")
X = df.drop('quality', axis=1).values
y = (df['quality'] < 6).values

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))


################################################################################
############### Ex. 11: Centering and scaling in a pipeline ####################
################################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))



################################################################################
####### Ex. 12: Bringing it all together I: Pipeline for classification ########
################################################################################
from sklearn.model_selection import GridSearchCV

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C': [1, 10, 100],
              'SVM__gamma': [0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3) #NB: Defaults to 3-fold cross-validation so optional

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


################################################################################
######## Ex. 13: Bringing it all together II: Pipeline for regression ##########
################################################################################
from sklearn.linear_model import ElasticNet
df = pd.read_csv("ml_techniques/01_supervised_scikit/_datasets/gapminder.csv")
y = df['life'].values
X = df.drop('life', axis=1).values

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters, cv=3) #NB: 3-fold cross-validation default

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))



################################################################################
################################### Run: #######################################
##### ipython ml_techniques/01_supervised_scikit/04_preprocessing/labs.py ######
################################################################################
