from sklearn.model_selection import train_test_split

# Convert lists of arrays to a single numpy array if not already
synthetic_price_paths_array = np.array(synthetic_price_paths)

# Define the split ratio (e.g., 80% for training, 20% for testing)
test_size_ratio = 0.2

# Split the synthetic price paths
S_train, S_test = train_test_split(synthetic_price_paths_array, test_size=test_size_ratio, random_state=42)

# Split the corresponding Black-Scholes prices
BS_prices_train, BS_prices_test = train_test_split(all_bs_prices, test_size=test_size_ratio, random_state=42)

# Split the corresponding Black-Scholes deltas
BS_deltas_train, BS_deltas_test = train_test_split(all_bs_deltas, test_size=test_size_ratio, random_state=42)

print(f"Training set size: {len(S_train)} paths")
print(f"Test set size: {len(S_test)} paths")

print(f"Shape of S_train: {S_train.shape}")
print(f"Shape of BS_deltas_train: {BS_deltas_train.shape}")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# --- Model Parameters ---
num_dense_units = 32 # Number of units in the single Dense layer

# --- Input Layers for a single time step ---
# Input for current asset price (S_t) - single value per time step
s_input_t = Input(shape=(1,), name='S_t_input')

# Input for Black-Scholes delta (BS_delta_t) - single value per time step
bs_delta_input_t = Input(shape=(1,), name='BS_delta_t_input')

# Input for previous delta position (delta_{t-1}) - single value per time step
prev_delta_input_t = Input(shape=(1,), name='prev_delta_input')

# --- Feature Concatenation ---
# Concatenate S_t, BS_delta_t, and prev_delta_t as features for the Dense layer
combined_features_t = Concatenate(axis=-1)([s_input_t, bs_delta_input_t, prev_delta_input_t])

# --- Single Dense Layer ---
# The Dense layer will process the combined features
dense_output = Dense(num_dense_units, activation='tanh')(combined_features_t)
output_delta_t = Dense(1, activation='tanh', name='delta_t_output')(dense_output) # Final output layer

# --- Create the Model for a single time step ---
# Renaming to `deep_hedger_model` to distinguish from a full sequence model
deep_hedger_model = Model(inputs=[s_input_t, bs_delta_input_t, prev_delta_input_t], outputs=output_delta_t)

deep_hedger_model.summary()

print("Deep Hedging model implemented as a single-step predictor, incorporating previous delta.")

import tensorflow as tf

# Assuming option_K, r, option_type, dt are already defined globally or passed
# target_strike, r, option_type, dt

@tf.function
def calculate_pnl(S_path_batch, deltas_batch, K, r_tf, option_type_str):
    """
    Calculates the P&L for a batch of price paths and corresponding delta hedges.

    Args:
        S_path_batch (tf.Tensor): Batch of underlying price paths (batch_size, num_steps + 1).
        deltas_batch (tf.Tensor): Batch of delta hedges (batch_size, num_steps + 1).
                                 deltas_batch[..., t] is the delta held from t to t+1.
        K (tf.Tensor): Strike price.
        r_tf (tf.Tensor): Risk-free rate.
        option_type_str (tf.Tensor): String tensor indicating 'call' or 'put'.

    Returns:
        tf.Tensor: A tensor of P&L values for each path in the batch (batch_size,).
    """
    # Ensure tensors are float32 for consistency
    S_path_batch = tf.cast(S_path_batch, tf.float32)
    deltas_batch = tf.cast(deltas_batch, tf.float32)
    K = tf.cast(K, tf.float32)
    r_tf = tf.cast(r_tf, tf.float32)

    # Reshape deltas_batch to (batch_size, num_steps + 1) if it's (batch_size, num_steps + 1, 1)
    if len(deltas_batch.shape) == 3: # Handle case where delta_output is (batch_size, timesteps, 1)
        deltas_batch = tf.squeeze(deltas_batch, axis=-1)

    # S_t is S_path_batch[:, :-1]
    # S_{t+1} is S_path_batch[:, 1:]
    price_changes = S_path_batch[:, 1:] - S_path_batch[:, :-1]

    # Delta position at time t is held from t to t+1. So we use deltas_batch[:, :-1] (or just deltas_batch) for trading P&L
    # The last delta in deltas_batch is the delta at expiry, which contributes to the final option value, not trading PnL
    # This matches the (delta . S)_T definition in the paper where delta_ti is for (S_{ti+1} - S_ti)
    trading_pnl_per_interval = deltas_batch[:, :-1] * price_changes
    cumulative_trading_pnl = tf.reduce_sum(trading_pnl_per_interval, axis=1)

    # Option payoff at expiry (Z_T)
    S_T = S_path_batch[:, -1] # Final underlying price

    # Check option type using tf.strings.equal
    is_call = tf.cast(tf.strings.bytes_split(option_type_str)[0] == tf.constant(b'c'), tf.float32)

    option_payoff_at_expiry_call = tf.maximum(0.0, S_T - K)
    option_payoff_at_expiry_put = tf.maximum(0.0, K - S_T)

    # Use tf.where to select based on option type
    option_payoff_at_expiry = tf.where(
        tf.cast(is_call, tf.bool),
        option_payoff_at_expiry_call,
        option_payoff_at_expiry_put
    )

    # Total P&L: -Z_T + cumulative_trading_pnl (no transaction costs)
    total_pnl = -option_payoff_at_expiry + cumulative_trading_pnl

    return total_pnl

@tf.function
def cvar_loss(pnl_values, alpha=0.05):
    """
    Calculates the CVaR (Conditional Value at Risk) loss for a batch of P&L values.
    CVaR is the expected loss in the worst 'alpha' percentile.

    Args:
        pnl_values (tf.Tensor): A batch of P&L values (batch_size,).
        alpha (float): The significance level for CVaR (e.g., 0.05 for 5%).

    Returns:
        tf.Tensor: The CVaR loss (scalar).
    """
    # Sort P&L values in ascending order (from worst loss to best profit)
    sorted_pnl = tf.sort(pnl_values)

    # Calculate the index for VaR_alpha (worst 'alpha' percentage)
    num_samples = tf.cast(tf.shape(sorted_pnl)[0], tf.float32)
    cvar_index = tf.cast(tf.floor(num_samples * alpha), tf.int32)

    # Select the worst 'alpha' percentile of P&L values
    # Note: tf.slice is exclusive of the end index, so we need to go up to cvar_index
    worst_pnl_values = sorted_pnl[:cvar_index + 1] # Include up to the alpha-th percentile

    # CVaR is the negative of the average of these worst P&L values
    # We want to minimize the loss, so we minimize -CVaR or maximize CVaR.
    # Conventionally, CVaR is expressed as a positive loss, so we take the negative mean of P&L
    # PNL is profit, so minimizing average PNL means making profits smaller.
    # If PNL is negative (a loss), then -PNL is positive, and minimizing it means minimizing the loss.
    # So, CVaR loss will be -mean(worst_pnl_values)
    cvar_value = -tf.reduce_mean(worst_pnl_values)

    return cvar_value

print("TensorFlow-compatible P&L calculation and CVaR loss functions defined.")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 100
batch_size = 32 # Can be adjusted

num_train_paths = S_train.shape[0]
num_time_steps = S_train.shape[1] # This includes S_0 to S_T

# Convert numpy arrays to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (S_train, BS_deltas_train)
).shuffle(num_train_paths).batch(batch_size)

print(f"Starting Deep Hedging Model Training for {epochs} epochs...")

for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    for step, (batch_S_paths, batch_BS_deltas) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Initialize an empty TensorArray to store predicted deltas for this batch
            # The size is num_time_steps because we predict delta_0 to delta_{T}
            batch_predicted_deltas_ta = tf.TensorArray(tf.float32, size=num_time_steps, dynamic_size=False, clear_after_read=False)

            # Initial delta_prev for each path in the batch (start with 0 hedge)
            delta_prev = tf.zeros((batch_S_paths.shape[0], 1), dtype=tf.float32)

            # Iterate through time steps to get predicted deltas for each path in the batch
            for t in tf.range(num_time_steps):
                # Prepare inputs for the deep_hedger_model for the current time step `t`
                S_t_batch = tf.expand_dims(batch_S_paths[:, t], axis=-1)         # (batch_size, 1)
                BS_delta_t_batch = tf.expand_dims(batch_BS_deltas[:, t], axis=-1) # (batch_size, 1)

                # Get the predicted delta for the current time step from the model
                predicted_delta_t = deep_hedger_model([S_t_batch, BS_delta_t_batch, delta_prev])

                # Store the predicted delta
                batch_predicted_deltas_ta = batch_predicted_deltas_ta.write(t, predicted_delta_t)

                # Update delta_prev for the next time step
                delta_prev = predicted_delta_t

            # Stack the predicted deltas from the TensorArray. Resulting shape: (num_time_steps, batch_size, 1)
            # Transpose to get (batch_size, num_time_steps, 1), then squeeze to (batch_size, num_time_steps)
            final_predicted_deltas_seq = tf.transpose(batch_predicted_deltas_ta.stack(), perm=[1, 0, 2])
            final_predicted_deltas_squeezed = tf.squeeze(final_predicted_deltas_seq, axis=-1) # (batch_size, num_time_steps)

            # Calculate P&L for the entire batch of paths using the predicted deltas
            pnl_values = calculate_pnl(
                batch_S_paths,
                final_predicted_deltas_squeezed,
                tf.constant(option_K, dtype=tf.float32),
                tf.constant(option_r, dtype=tf.float32),
                tf.constant(option_type, dtype=tf.string)
            )

            # Calculate the CVaR loss from the P&L values
            loss = cvar_loss(pnl_values, alpha=0.05)

        # Compute and apply gradients
        gradients = tape.gradient(loss, deep_hedger_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, deep_hedger_model.trainable_variables))

        epoch_loss_avg.update_state(loss)

    print(f"Epoch {epoch+1}/{epochs}, CVaR Loss: {epoch_loss_avg.result():.4f}")

print("\nDeep Hedging Model Training Complete.")

# Helper function to predict deltas for a given set of paths
@tf.function
def predict_deltas_for_paths(model, S_paths, BS_deltas_input, num_time_steps):
    all_predicted_deltas = tf.TensorArray(tf.float32, size=S_paths.shape[0], dynamic_size=False, clear_after_read=False)

    for i in tf.range(S_paths.shape[0]): # Iterate over each path in the batch
        s_path = S_paths[i]
        bs_deltas_path = BS_deltas_input[i]

        path_predicted_deltas_ta = tf.TensorArray(tf.float32, size=num_time_steps, dynamic_size=False, clear_after_read=False)
        delta_prev = tf.zeros((1, 1), dtype=tf.float32) # Initial delta for a single path

        for t in tf.range(num_time_steps):
            S_t = tf.expand_dims(s_path[t], axis=0) # (1, 1)
            BS_delta_t = tf.expand_dims(bs_deltas_path[t], axis=0) # (1, 1)

            predicted_delta_t = model([S_t, BS_delta_t, delta_prev])
            path_predicted_deltas_ta = path_predicted_deltas_ta.write(t, predicted_delta_t)
            delta_prev = predicted_delta_t

        final_path_predicted_deltas = tf.squeeze(path_predicted_deltas_ta.stack(), axis=-1) # (num_time_steps,)
        all_predicted_deltas = all_predicted_deltas.write(i, final_path_predicted_deltas)

    return all_predicted_deltas.stack()

print("Calculating P&L for Deep Hedger on Train Set...")
# Deep Hedger P&L on Train Set
dh_predicted_deltas_train = predict_deltas_for_paths(
    deep_hedger_model, S_train, BS_deltas_train, num_time_steps
)
dh_pnl_train = calculate_pnl(
    S_train,
    dh_predicted_deltas_train,
    tf.constant(option_K, dtype=tf.float32),
    tf.constant(option_r, dtype=tf.float32),
    tf.constant(option_type, dtype=tf.string)
).numpy() # Convert to numpy for analysis

print("Calculating P&L for Deep Hedger on Test Set...")
# Deep Hedger P&L on Test Set
dh_predicted_deltas_test = predict_deltas_for_paths(
    deep_hedger_model, S_test, BS_deltas_test, num_time_steps
)
dh_pnl_test = calculate_pnl(
    S_test,
    dh_predicted_deltas_test,
    tf.constant(option_K, dtype=tf.float32),
    tf.constant(option_r, dtype=tf.float32),
    tf.constant(option_type, dtype=tf.string)
).numpy() # Convert to numpy for analysis

print("Deep Hedger P&L calculated for train and test sets.")

print("Calculating P&L for Black-Scholes Delta Hedging on Train Set...")

# Black-Scholes Delta Hedging P&L on Train Set
bs_pnl_train = []
for path_idx in range(S_train.shape[0]):
    s_path = S_train[path_idx]
    # Use the pre-calculated Black-Scholes deltas directly
    bs_deltas_for_path_tf = tf.constant(BS_deltas_train[path_idx], dtype=tf.float32)

    pnl = calculate_pnl(
        tf.expand_dims(tf.constant(s_path, dtype=tf.float32), axis=0),
        tf.expand_dims(bs_deltas_for_path_tf, axis=0),
        tf.constant(option_K, dtype=tf.float32),
        tf.constant(option_r, dtype=tf.float32),
        tf.constant(option_type, dtype=tf.string)
    ).numpy()[0]
    bs_pnl_train.append(pnl)
bs_pnl_train = np.array(bs_pnl_train)

print("Calculating P&L for Black-Scholes Delta Hedging on Test Set...")

# Black-Scholes Delta Hedging P&L on Test Set
bs_pnl_test = []
for path_idx in range(S_test.shape[0]):
    s_path = S_test[path_idx]
    # Use the pre-calculated Black-Scholes deltas directly
    bs_deltas_for_path_tf = tf.constant(BS_deltas_test[path_idx], dtype=tf.float32)

    pnl = calculate_pnl(
        tf.expand_dims(tf.constant(s_path, dtype=tf.float32), axis=0),
        tf.expand_dims(bs_deltas_for_path_tf, axis=0),
        tf.constant(option_K, dtype=tf.float32),
        tf.constant(option_r, dtype=tf.float32),
        tf.constant(option_type, dtype=tf.string)
    ).numpy()[0]
    bs_pnl_test.append(pnl)
bs_pnl_test = np.array(bs_pnl_test)

print("Black-Scholes Delta Hedging P&L calculated for train and test sets.")

def calculate_var(pnl_values, alpha=0.05):
    """
    Calculates Value at Risk (VaR) for a given set of P&L values.
    """
    pnl_values_sorted = np.sort(pnl_values)
    index = int(np.floor(len(pnl_values_sorted) * alpha))
    return pnl_values_sorted[index]

# Convert the tf.function cvar_loss to a numpy-compatible function for reporting
def calculate_cvar_numpy(pnl_values, alpha=0.05):
    return cvar_loss(tf.constant(pnl_values, dtype=tf.float32), alpha=alpha).numpy()

print("VaR and NumPy-compatible CVaR calculation functions defined.")

import matplotlib.pyplot as plt
import seaborn as sns

alph_level = 0.05

# --- Plotting P&L Distributions ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle('P&L Distributions for Deep Hedging vs. Black-Scholes Delta Hedging', fontsize=16)

sns.histplot(dh_pnl_train, bins=50, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Deep Hedger (Train Set)')
axes[0, 0].set_xlabel('P&L')
axes[0, 0].set_ylabel('Frequency')

sns.histplot(bs_pnl_train, bins=50, kde=True, ax=axes[0, 1], color='lightcoral')
axes[0, 1].set_title('Black-Scholes Delta Hedger (Train Set)')
axes[0, 1].set_xlabel('P&L')
axes[0, 1].set_ylabel('Frequency')

sns.histplot(dh_pnl_test, bins=50, kde=True, ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Deep Hedger (Test Set)')
axes[1, 0].set_xlabel('P&L')
axes[1, 0].set_ylabel('Frequency')

sns.histplot(bs_pnl_test, bins=50, kde=True, ax=axes[1, 1], color='lightcoral')
axes[1, 1].set_title('Black-Scholes Delta Hedger (Test Set)')
axes[1, 1].set_xlabel('P&L')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Reporting VaR and CVaR ---
print("\n--- Hedging Performance Metrics (alpha = {alph_level:.0%}) ---")

# Deep Hedger Train
mean_pnl_dh_train = np.mean(dh_pnl_train)
std_pnl_dh_train = np.std(dh_pnl_train)
var_dh_train = calculate_var(dh_pnl_train, alpha=alph_level)
cvar_dh_train = calculate_cvar_numpy(dh_pnl_train, alpha=alph_level)
print(f"\nDeep Hedger (Train Set):\n  Mean P&L: {mean_pnl_dh_train:.2f}\n  Std Dev P&L: {std_pnl_dh_train:.2f}\n  VaR ({alph_level:.0%}): {var_dh_train:.2f}\n  CVaR ({alph_level:.0%}): {cvar_dh_train:.2f}")

# Black-Scholes Hedger Train
mean_pnl_bs_train = np.mean(bs_pnl_train)
std_pnl_bs_train = np.std(bs_pnl_train)
var_bs_train = calculate_var(bs_pnl_train, alpha=alph_level)
cvar_bs_train = calculate_cvar_numpy(bs_pnl_train, alpha=alph_level)
print(f"\nBlack-Scholes Delta Hedger (Train Set):\n  Mean P&L: {mean_pnl_bs_train:.2f}\n  Std Dev P&L: {std_pnl_bs_train:.2f}\n  VaR ({alph_level:.0%}): {var_bs_train:.2f}\n  CVaR ({alph_level:.0%}): {cvar_bs_train:.2f}")

# Deep Hedger Test
mean_pnl_dh_test = np.mean(dh_pnl_test)
std_pnl_dh_test = np.std(dh_pnl_test)
var_dh_test = calculate_var(dh_pnl_test, alpha=alph_level)
cvar_dh_test = calculate_cvar_numpy(dh_pnl_test, alpha=alph_level)
print(f"\nDeep Hedger (Test Set):\n  Mean P&L: {mean_pnl_dh_test:.2f}\n  Std Dev P&L: {std_pnl_dh_test:.2f}\n  VaR ({alph_level:.0%}): {var_dh_test:.2f}\n  CVaR ({alph_level:.0%}): {cvar_dh_test:.2f}")

# Black-Scholes Hedger Test
mean_pnl_bs_test = np.mean(bs_pnl_test)
std_pnl_bs_test = np.std(bs_pnl_test)
var_bs_test = calculate_var(bs_pnl_test, alpha=alph_level)
cvar_bs_test = calculate_cvar_numpy(bs_pnl_test, alpha=alph_level)
print(f"\nBlack-Scholes Delta Hedger (Test Set):\n  Mean P&L: {mean_pnl_bs_test:.2f}\n  Std Dev P&L: {std_pnl_bs_test:.2f}\n  VaR ({alph_level:.0%}): {var_bs_test:.2f}\n  CVaR ({alph_level:.0%}): {cvar_bs_test:.2f}")
