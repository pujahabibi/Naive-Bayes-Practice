import pandas as pd
import numpy as np

# Create an empty dataframe
data = pd.DataFrame()

# Create our target variable
data['Defective'] = ['No', 'No', 'No', 'Yes', 'Yes']

# Create our feature variables
data['Branch'] = [5, 3, 9, 15, 16]
data['LOC'] = [15, 5, 20, 40, 35]

# View the data
print(data)

test_defect = pd.DataFrame()

test_defect['Branch'] = [16]
test_defect['LOC'] = [39]

print()
print(test_defect)

n_defective = data['Defective'][data['Defective'] == 'Yes'].count()
n_non_defective = data['Defective'][data['Defective'] == 'No'].count()
total_defect = data['Defective'].count()

# Number of males divided by the total rows
P_defective = n_defective/total_defect

# Number of females divided by the total rows
P_non_defective = n_non_defective/total_defect

# Group the data by gender and calculate the means of each feature
data_means = data.groupby('Defective').mean()

# View the values
print()
print("--------- Data Means -----------")
print(data_means)

# Group the data by gender and calculate the variance of each feature
data_variance = data.groupby('Defective').var()

# View the values
print()
print("-------- Data Variance ----------")
print(data_variance)

# Means for male
defective_bc_mean = data_means['Branch'][data_variance.index == 'Yes'].values[0]
defective_loc_mean = data_means['LOC'][data_variance.index == 'Yes'].values[0]

# Variance for male
defective_bc_variance = data_variance['Branch'][data_variance.index == 'Yes'].values[0]
defective_loc_variance = data_variance['LOC'][data_variance.index == 'Yes'].values[0]

# Means for female
non_defective_bc_mean = data_means['Branch'][data_variance.index == 'No'].values[0]
non_defective_loc_mean = data_means['LOC'][data_variance.index == 'No'].values[0]

# Variance for female
non_defective_bc_variance = data_variance['Branch'][data_variance.index == 'No'].values[0]
non_defective_loc_variance = data_variance['LOC'][data_variance.index == 'No'].values[0]

print()
print(defective_bc_mean, defective_loc_mean, defective_bc_variance, defective_loc_variance)

# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))

    # return p
    return p

print()
a = P_defective *p_x_given_y(test_defect['Branch'][0], defective_bc_mean, defective_bc_variance) * \
    p_x_given_y(test_defect['LOC'][0], defective_bc_mean, defective_bc_variance)

b = P_non_defective * p_x_given_y(test_defect['Branch'][0], non_defective_bc_mean, non_defective_bc_variance) *\
    p_x_given_y(test_defect['LOC'][0], non_defective_bc_mean, non_defective_bc_variance)

if a > b:
    print("Defect")
else:
    print("No-Defect")