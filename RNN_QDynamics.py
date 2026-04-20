import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

## The physical system we're experimenting with here is photons
## Though it seems that by schrodinger's equation it's not limited
## we could have vibration, photon, electrons, spins -- very cool

# This is the experimental timeframe:
t_final = 4.0
steps = 100
tlist = np.linspace(0, t_final, steps+1)

# Physical Parameters
# In QuTiP, we typically set Planck's constant (hbar) = 1 for simplicity.
Omega_R = 2 * np.pi * 0.82 # Rabi frequency
gamma_m = 0.40
gamma_phi = 1.1

# Define the Qubit Operators
# Pauli matrices are the fundamental building blocks of qubit logic.
sig_x = qt.sigmax()
sig_y = qt.sigmay()
sig_z = qt.sigmaz()

# Set the Initial State
# A qubit perfectly in the ground state |0>
psi0 = qt.basis(2, 0)


# Build the Hamiltonian (The Microwave Drive)
# H_R = (\hbar \Omega_R / 2) * \sigma_X
H = 0.5 * Omega_R * sig_x

c_ops = [np.sqrt(gamma_phi - gamma_m)*sig_z]

# Stochastic collapse, this is the continuous weak measurement on the Z-axis
# This dictates teh random "jumps" cause by observing the system.
sc_ops = [np.sqrt(gamma_m)*sig_z]

# 7. Run the Stochastic Master Equstion (SME) solver
# Asking QuTip to evolve the system using the smesolve function 
print("Running Quantum Simulation...")
result = qt.smesolve(
    H,             # The drive
    psi0,          # The starting state
    tlist,         # The 4us timeline
    c_ops=c_ops,   # Normal noise
    sc_ops=sc_ops, # Measurement back-action
    e_ops=[sig_x, sig_y, sig_z], # What we want to track (expectation values)
    ntraj=1        # We only want to generate 1 specific trajectory right now
)

# 8. Extract the Noisy Measurement Record (V_t)
# In QuTiP, the stochastic solver stores the continuous measurement noise in the 'wiener' process.
# We extract this to represent the raw voltage output from the parametric amplifier.
V_t = result.wiener_process[0][0]

print(f"Simulation complete. Generated {len(V_t)} data points.")




# Step 1: Tech Stack
# Requier: PyTorch or TensorFlow what ever is preffered
# Quatum Simulation: QuTiP
# Step 2: Simulating the Data (Your Virtual Quantum Fridge)
# Need to simulate a single qubit being being driven by a microwave pulse
# (The Rabi Hamiltonian)
# Whilst continously being subjected to a weak-measurement along the Z-Axis
# For each trace we need two outputs: 1) The continuous, 
# noisy measurement record (V t) sampled in 40 ns time steps over a total of 4 μs 
# The final, definitive "projective" measurement outcome (either a 0 or a 1) at the end of the 4 μs
#