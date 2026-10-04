import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# Load the 20 different parameter sets from the Excel file
params_df = pd.read_excel('20_dif_val.xlsx')

# Load the variance data for comparison
variance_df = pd.read_excel('r_alpha_Est.xlsx')
variance = variance_df['Variance'].values

# Initialize lists to store results
alpha_solutions = []
r_values = []

# Loop through each parameter set
for index, row in params_df.iterrows():
    # Retrieve parameters for the current run
    ra = row['r'] / row['alpha']  # Ensure ra is consistent with the parameter set
    s = row['s']
    B = row['B']
    w = 0
    TM = 10 * round(B / s)
    T = np.arange(1, TM + 1)
    
    def variance_eq(alpha):
        calculated_variance = (
    ra * B / (s - 1) * (1 - (B / (B + T - w))**(s - 1))
    + ra * w
    - (ra**2) * ((B / (s - 1)) * (1 - (B / (B + T - w))**(s - 1)) + w)**2
    + (2 * ra * (ra + 1) * B / (alpha * (s - 1))) * (
        (B / (s - 2) - (B / (B + T - w))**(s - 2)) - T * (B / (B + T - w))**(s - 1) + w
    )
    + (ra * (ra + 1) / alpha) * w**2
)
        return np.mean(calculated_variance)

    # Initial guess for alpha
    alpha_initial_guess = 6.0
    alpha_solution = fsolve(variance_eq, alpha_initial_guess)[0]

    # Calculate corresponding r value
    r_value = ra * alpha_solution

    # Append results to lists
    alpha_solutions.append(alpha_solution)
    r_values.append(r_value)

    print(f"Alpha for parameter set {index + 1} calculated.")

# Add results to the DataFrame
params_df['calculated_alpha'] = alpha_solutions
params_df['calculated_r'] = r_values

# Save the updated DataFrame back to the existing sheet
params_df.to_excel('20_dif_val.xlsx', sheet_name='20_def_val', index=False)

print("All results saved successfully to the sheet '20_def_val' in the file '20_dif_val.xlsx'.")