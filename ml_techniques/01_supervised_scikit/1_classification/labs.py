import os

# Course deps
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# Open shared user defined functions
udf = "ml_techniques/udfs.py"
exec(open(os.path.abspath(udf)).read())

# Set seed used in exercises
# np.random.seed(42)

################################################################################
################################## Ex. 3 #######################################
################################################################################

iris = datasets.load_iris()
iris.keys()
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
type(iris.data)
# numpy.ndarray
iris.data.shape
# (150, 4)

X  = iris.data
y  = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df.head()
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# 3                4.6               3.1                1.5               0.2
# 4                5.0               3.6                1.4               0.2

_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')
plt.show()


################################################################################
########################### Ex. 4: Numerical EDA ###############################
################################################################################

# Setup data
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
df.replace(to_replace='?', value=np.nan, inplace=True)
df.fillna(df.mean().round(0).astype(int), inplace=True)
df[column_names[1:]] = df[column_names[1:]].astype(int)

# Inspect data
df.head()
df.info()
df.describe()

################################################################################
########################### Ex. 4: Graphical EDA ###############################
################################################################################

plt.figure() # Start new plot
sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

plt.figure() # Start new plot
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

################################################################################
################################### Run: #######################################
##### ipython ml_techniques/01_supervised_scikit/1_classification/labs.py ######
################################################################################



