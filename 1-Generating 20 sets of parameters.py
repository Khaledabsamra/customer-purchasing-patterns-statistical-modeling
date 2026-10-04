import numpy as np
import pandas as pd

# Set number of samples
num_samples = 20

# Initialize lists to store the parameters
r_values = []
alpha_values = []
s_values = []
B_values = []

# Generate random values with constraints
while len(r_values) < num_samples:
    r = np.random.uniform(1, 10)
    alpha = np.random.uniform(1, 10)
    B = np.random.uniform(1, 10)
    s = np.random.uniform(1, 10)
    
    # Ensure r > alpha and B > s
    if r > alpha and B > s:
        r_values.append(r)
        alpha_values.append(alpha)
        s_values.append(s)
        B_values.append(B)

# Combine values into a DataFrame
parameters_df = pd.DataFrame({
    'r': r_values,
    'alpha': alpha_values,
    's': s_values,
    'B': B_values
})

# Calculate ra as r / alpha and add it as a new column
parameters_df['ra'] = parameters_df['r'] / parameters_df['alpha']

# Save the DataFrame to an Excel file named '20_dif_val.xlsx' including the 'ra' column
parameters_df.to_excel("20_dif_val.xlsx", index=False)

print("Values successfully saved to 20_dif_val.xlsx.")