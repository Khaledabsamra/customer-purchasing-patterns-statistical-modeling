## parameters estsimation for car data where w is measured by seasons ##
import numpy as np
import pandas as pd
import math

TM = 25  # according to the graph it's best suitable
num_samples = 50  # Number of different points used in gradient descent
iterations = 10000  # Maximum number of iterations
tolerance = 1e-6

# Read the Excel file into a DataFrame
df = pd.read_excel('Arranged_shifted_Data_season.xlsx')

X = df['X'].values
w = df['W_seasons'].values[0]
T = np.arange(w, int(TM) + 1)
customer_purchase_counts = []

singleCustomerArray = [np.sum((np.array(X) <= j + 1)) for j in T]
customer_purchase_counts.append(singleCustomerArray)

# Calculate cumulative means
X_bar_T = np.mean(customer_purchase_counts, axis=0)
# Generate random values for ra, s, and B
ra_samples = np.random.rand(num_samples) 
s_samples = np.random.rand(num_samples) 
B_samples = np.random.rand(num_samples) 

# Combine them into an array of tuples
samples = np.array(list(zip(ra_samples, s_samples, B_samples)))

# Function to compute the function value
def function_value(ra, s, B, T, X_bar_T):
    try:
        return np.sum(((ra * ((B) / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w) - X_bar_T) ** 2)
    except Exception as e:
        print(f"Error in function_value: {e}")
        return np.nan

# Function to compute the gradients
def compute_gradients(ra, s, B, T, X_bar_T):
    try:
        G = 2 * (ra * ((B / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w) - X_bar_T)
        grad_ra = np.sum(G * ((B / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w))
        grad_s = np.sum(G * ((-(ra * B) / ((s - 1) ** 2)) * (1 - np.power(B / (B + T - w), (s - 1))) -
                             (((ra * B) / (s - 1)) * np.power(B / (B + T - w), (s - 1)) * np.log(B / (B + T - w)))))
        grad_B = np.sum(G * ((ra / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) -
                             ((ra * (T - w)) * np.power(B, s - 1)) / (np.power(B + T - w, s))))

        # Normalize gradients if they are too large
        norm = np.linalg.norm([grad_ra, grad_s, grad_B])
        if norm > 1:
            grad_ra /= norm
            grad_s /= norm
            grad_B /= norm

        return grad_ra, grad_s, grad_B
    except Exception as e:
        print(f"Error in compute_gradients: {e}")
        return np.nan, np.nan, np.nan

# Adam optimizer
def Adam_optimizer(ra, s, B, T, cumulative_mean_purchase_counts):
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m_ra, m_s, m_B = 0, 0, 0
    v_ra, v_s, v_B = 0, 0, 0
    t = 0

    for iteration in range(iterations):
        t += 1
        grad_ra, grad_s, grad_B = compute_gradients(ra, s, B, T, cumulative_mean_purchase_counts)

        m_ra = beta1 * m_ra + (1 - beta1) * grad_ra
        m_s = beta1 * m_s + (1 - beta1) * grad_s
        m_B = beta1 * m_B + (1 - beta1) * grad_B

        v_ra = beta2 * v_ra + (1 - beta2) * (grad_ra ** 2)
        v_s = beta2 * v_s + (1 - beta2) * (grad_s ** 2)
        v_B = beta2 * v_B + (1 - beta2) * (grad_B ** 2)

        m_ra_hat = m_ra / (1 - beta1 ** t)
        m_s_hat = m_s / (1 - beta1 ** t)
        m_B_hat = m_B / (1 - beta1 ** t)

        v_ra_hat = v_ra / (1 - beta2 ** t)
        v_s_hat = v_s / (1 - beta2 ** t)
        v_B_hat = v_B / (1 - beta2 ** t)

        ra = max(ra - alpha * m_ra_hat / (np.sqrt(v_ra_hat) + epsilon), 0.1)
        s = max(s - alpha * m_s_hat / (np.sqrt(v_s_hat) + epsilon), 1.01)
        B = max(B - alpha * m_B_hat / (np.sqrt(v_B_hat) + epsilon), 0.1)

        func_value = function_value(ra, s, B, T, X_bar_T)
        if np.isnan(func_value):
            return None, None, None

    return ra, s, B

# Initialize variables to store the best result
best_func_value = np.inf
best_params = None

# Prepare columns to store results
estimated_ra_values = []
estimated_s_values = []
estimated_B_values = []

# Run Adam optimizer
best_func_value = np.inf
best_params = None

for _ in range(num_samples):
    ra_init, s_init, B_init = np.random.uniform(1, 10, 3)
    final_ra, final_s, final_B = Adam_optimizer(ra_init, s_init, B_init, T, X_bar_T)

    if final_ra is None or final_s is None or final_B is None:
        continue

    final_func_value = function_value(final_ra, final_s, final_B, T, X_bar_T)

    if final_func_value < best_func_value:
        best_func_value = final_func_value
        best_params = (final_ra, final_s, final_B)

# Store the best results
if best_params:
    estimated_ra_values = [best_params[0]]
    estimated_s_values = [best_params[1]]
    estimated_B_values = [best_params[2]]
else:
    estimated_ra_values = [None]
    estimated_s_values = [None]
    estimated_B_values = [None]
    
# Add results to the dataframe and save

if best_params:
    estimated_ra_values = [best_params[0]] * len(df)  # Repeat the value for each row
    estimated_s_values = [best_params[1]] * len(df)  # Repeat the value for each row
    estimated_B_values = [best_params[2]] * len(df)  # Repeat the value for each row
else:
    estimated_ra_values = [None] * len(df)  # Repeat None for each row
    estimated_s_values = [None] * len(df)  # Repeat None for each row
    estimated_B_values = [None] * len(df)  # Repeat None for each row

# Add the replicated estimated values to the dataframe
df['Estimated_ra'] = estimated_ra_values
df['Estimated_s'] = estimated_s_values
df['Estimated_B'] = estimated_B_values

# Save the updated dataframe to an Excel file
df.to_excel('parameters_car_season_2.xlsx', index=False)
print("Estimated Values successfully saved to parameters_car_season_2.xlsx")