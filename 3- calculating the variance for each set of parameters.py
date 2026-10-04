import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from numpy import exp

# Load the 20 different parameter sets from the Excel file
params_df = pd.read_excel('20_dif_val.xlsx')

# Define constants
N = 1000  # Number of customers

# Function for exponential distribution
def exp(L):
    return np.random.exponential(1 / L)

# Function for gamma distribution with inverse scale (1/B)
def gamma(s, B):
    return np.random.gamma(s, 1 / B)  # Scale parameter should be 1/B

# Initialize an Excel writer to save all results in one file
output_file = 'r_alpha_Est.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

    # Loop through each parameter set
    for index, row in params_df.iterrows():
        # Retrieve parameters for the current run
        r = row['r']
        alpha = row['alpha']
        s = row['s']
        B = row['B']
        ra = r / alpha
        TM = 10 * round(B / s)

        # Initialize lists to store data
        X_values = []
        t_values = []
        Tau_values = []
        customer_purchase_counts = []

        # Run simulation for N customers
        for i in range(N):
            # Generate customer lifetime from gamma distribution
            mu = gamma(s, B)
            Tau = exp(mu)

            # Generate Lambda from gamma distribution
            lam = gamma(r, alpha)

            # Generate purchase times
            current_time = 0
            purchase_times = []
            singleCustomerArray = []

            while current_time <= min(Tau, TM):
                current_time += exp(lam)
                if current_time < min(Tau, TM):
                    purchase_times.append(current_time)

            if purchase_times:
                X_values.append(len(purchase_times))
                t_values.append(purchase_times[-1])
                Tau_values.append(Tau)

                for j in range(TM):
                    singleCustomerArray.append(np.sum((np.array(purchase_times) <= j + 1)))

                customer_purchase_counts.append(singleCustomerArray)
            else:
                customer_purchase_counts.append([0] * TM)
                X_values.append(0)
                t_values.append(0)
                Tau_values.append(Tau)

        # Compute the variance across customers for each interval
        variance = np.var(customer_purchase_counts, axis=0)

        # Convert variance to DataFrame
        variance_df = pd.DataFrame(variance, columns=['Variance'])

        # Save each variance result to a separate sheet named by the parameter set index
        sheet_name = f'Set_{index + 1}'
        variance_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Variance for parameter set {index + 1} saved to sheet '{sheet_name}'.")

print(f"All variances saved successfully to r_alpha_Est.")