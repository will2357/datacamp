import os
udf = "02_python_stats_2/1_4_old_labs/user_defined_functions.py"
exec(open(os.path.abspath(udf)).read())

################################################################################
############################# Data for Ex. 2 ###################################
################################################################################

df1 = pd.read_csv("02_python_stats_2/_datasets/finch_beaks_1975.csv")
df1['year'] = '1975'
df1.rename(columns = {'Beak depth, mm':'beak_depth'}, inplace = True)
df1_final = df1[['beak_depth', 'year']]

df2 = pd.read_csv("02_python_stats_2/_datasets/finch_beaks_2012.csv")
df2['year'] = '2012'
df2.rename(columns = {'bdepth':'beak_depth'}, inplace = True)
df2_final = df2[['beak_depth', 'year']]
df = pd.concat([df1_final, df2_final])

################################################################################
############## Ex. 2: EDA of beak depths of Darwin's finches ###################
################################################################################

# Create bee swarm plot
_ = sns.swarmplot(data=df, x='year', y='beak_depth')

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()

################################################################################
############################# Data for Ex. 3 ###################################
################################################################################

bd_1975 = df1_final['beak_depth'].values
bd_2012 = df2_final['beak_depth'].values

################################################################################
####################### Ex. 3: ECDFs of beak depths ############################
################################################################################

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()






















################################################################################
############### ipython 02_python_stats_2/5_case_study/labs.py #################
################################################################################



