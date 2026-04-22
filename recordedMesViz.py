import torch
import matplotlib.pyplot as plt

# Load a training file
X_train = torch.load('X_train.pt') # Shape: [Num_Traces, Time_Steps, 1]

# Inspect the first trace (index 0)
sample_trace = X_train[1].flatten().numpy()

plt.figure(figsize=(10, 4))
plt.plot(sample_trace, color='blue', alpha=0.7)
plt.title("Quantum Measurement Record ($V_t$)")
plt.xlabel("Time Steps (40ns each)")
plt.ylabel("Voltage (Normalized)")
plt.show()