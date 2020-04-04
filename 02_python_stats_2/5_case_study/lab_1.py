import os
udf = "02_python_stats_2/1_4_old_labs/user_defined_functions.py"
exec(open(os.path.abspath(udf)).read())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1 = pd.read_csv("02_python_stats_2/_datasets/finch_beaks_1975.csv")
df1['year'] = '1975'
df1.rename(columns = {'Beak depth, mm':'beak_depth'}, inplace = True)
df1[['beak_depth', 'year']]

df2 = pd.read_csv("02_python_stats_2/_datasets/finch_beaks_2012.csv")
df2['year'] = '2012'
df2.rename(columns = {'bdepth':'beak_depth'}, inplace = True)
df2[['beak_depth', 'year']]

df = pd.concat([df1[['beak_depth', 'year']], df2[['beak_depth', 'year']]])

# Create bee swarm plot
_ = sns.swarmplot(data=df, x='year', y='beak_depth')

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()
