import numpy as np
import pandas as pd

# Parameters for the NBD/Pareto model
N = 15000  # Number of customers
correctionfactor = 1
num_samples = 100
iterations = 10000  # Maximum number of iterations
tolerance = 1e-6
w = 0

# Functions
def exp(L):
    return np.random.exponential(1 / L)

def gamma(s, B):
    return np.random.gamma(s, 1 / B)

# Optimization functions
def function_value(ra, s, B, T, X_bar_T):
    return np.sum((ra * ((B / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w) - X_bar_T) ** 2)

def compute_gradients(ra, s, B, T, X_bar_T):
    G = 2 * (ra * ((B / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w) - X_bar_T)
    grad_ra = np.sum(G * ((B / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) + w))
    grad_s = np.sum(G * ((-(ra * B) / ((s - 1) ** 2)) * (1 - np.power(B / (B + T - w), (s - 1))) -
                        (((ra * B) / (s - 1)) * np.power(B / (B + T - w), (s - 1)) * np.log(B / (B + T - w)))))
    grad_B = np.sum(G * ((ra / (s - 1)) * (1 - np.power(B / (B + T - w), (s - 1))) -
                         ((ra * (T - w)) * np.power(B, s - 1)) / (np.power(B + T - w, s))))
    return grad_ra, grad_s, grad_B

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

        func_value = function_value(ra, s, B, T, cumulative_mean_purchase_counts)
        if np.isnan(func_value):
            return None, None, None

    return ra, s, B

# Load data from Excel
df = pd.read_excel('20_dif_val.xlsx')

# Prepare columns to store results
estimated_ra_values = []
estimated_s_values = []
estimated_B_values = []

# Loop through each row in the dataframe
for idx, row in df.iterrows():
    r = row['r']
    alpha = row['alpha']
    s = row['s']
    B = row['B']

    ra = r / alpha
    TM = 10 * round(B / s)

    # Simulate customer data
    X_values = []
    t_values = []
    Tau_values = []
    customer_purchase_counts = []

    for i in range(N):
        mu = gamma(s, B)
        Tau = exp(mu)
        lam = gamma(r, alpha)

        current_time = 0
        purchase_times = []

        while current_time <= min(Tau, TM):
            current_time += exp(lam)
            if current_time < min(Tau, TM):
                purchase_times.append(current_time)

        if purchase_times:
            X_values.append(len(purchase_times))
            t_values.append(purchase_times[-1])
            Tau_values.append(Tau)

            singleCustomerArray = [np.sum((np.array(purchase_times) <= j + 1)) for j in range(TM)]
            customer_purchase_counts.append(singleCustomerArray)
        else:
            customer_purchase_counts.append([0] * TM)
            X_values.append(0)
            t_values.append(0)
            Tau_values.append(Tau)

    # Calculate cumulative means and variances
    cumulative_mean_purchase_counts = np.mean(customer_purchase_counts, axis=0)
    T = np.arange(1, int(TM) + 1)

    # Run Adam optimizer
    best_func_value = np.inf
    best_params = None

    for _ in range(num_samples):
        ra_init, s_init, B_init = np.random.uniform(1, 10, 3)
        final_ra, final_s, final_B = Adam_optimizer(ra_init, s_init, B_init, T, cumulative_mean_purchase_counts)

        if final_ra is None or final_s is None or final_B is None:
            continue

        final_func_value = function_value(final_ra, final_s, final_B, T, cumulative_mean_purchase_counts)

        if final_func_value < best_func_value:
            best_func_value = final_func_value
            best_params = (final_ra, final_s, final_B)

    if best_params:
        estimated_ra_values.append(best_params[0])
        estimated_s_values.append(best_params[1])
        estimated_B_values.append(best_params[2])
    else:
        estimated_ra_values.append(None)
        estimated_s_values.append(None)
        estimated_B_values.append(None)

# Add results to the dataframe and save
df['Estimated_ra'] = estimated_ra_values
df['Estimated_s'] = estimated_s_values
df['Estimated_B'] = estimated_B_values

df.to_excel('20_dif_val.xlsx', index=False)
print("Estimated Values successfully saved to 20_dif_val.xlsx.")