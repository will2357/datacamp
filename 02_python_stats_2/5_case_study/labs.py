import os
udf = "02_python_stats_2/1_4_old_labs/user_defined_functions.py"
exec(open(os.path.abspath(udf)).read())

# Set seed used in exercises
np.random.seed(42)

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
################ Ex. 4: Parameter estimates of beak depths #####################
################################################################################

# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_1975) - np.mean(bd_2012)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')


################################################################################
############# Ex. 5: Hypothesis test: Are beaks deeper in 2012? ################
################################################################################

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, 10_000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, 10_000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)

################################################################################
############################# Data for Ex. 7 ###################################
################################################################################

df1.rename(columns = {'Beak length, mm':'beak_length'}, inplace = True)
df2.rename(columns = {'blength':'beak_length'}, inplace = True)
bl_1975 = df1['beak_length'].values
bl_2012 = df2['beak_length'].values

################################################################################
################### Ex. 7: EDA of beak length and depth ########################
################################################################################

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()

################################################################################
####################### Ex. 8: Linear regressions ##############################
################################################################################

# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(
  bl_1975, bd_1975, 1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(
  bl_2012, bd_2012, 1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
intercept_conf_int_1975 = np.percentile(
  bs_intercept_reps_1975, [2.5, 97.5])
intercept_conf_int_2012 = np.percentile(
  bs_intercept_reps_2012, [2.5, 97.5])

# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)


################################################################################
############# Ex. 9: Displaying the linear regression results ##################
################################################################################

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
  plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
           linewidth=0.5, alpha=0.2, color='blue')
  plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
           linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()


################################################################################
#################### Ex. 10: Beak length to depth ratio ########################
################################################################################

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)








################################################################################
############### ipython 02_python_stats_2/5_case_study/labs.py #################
################################################################################



