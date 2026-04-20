import qutip as qt
import numpy as np
import torch

class QuantumDataGenerator:
    def __init__(self):
        """
        Initializes the virtual quantum laboratory with the physical parameters 
        and timeframe constraints extracted from the experiment.
        """
        # 1. THE EXPERIMENTAL TIMEFRAME
        self.t_final = 4.0
        self.steps = 100
        self.tlist = np.linspace(0, self.t_final, self.steps + 1)

        # 2. PHYSICAL PARAMETERS
        self.Omega_R = 2 * np.pi * 0.82
        self.gamma_m = 0.40
        self.gamma_phi = 1.1

        # 3. QUBIT LOGIC GATES
        self.sig_x = qt.sigmax()
        self.sig_y = qt.sigmay()
        self.sig_z = qt.sigmaz()

        # 4. INITIAL STATE (Ground state |0>)
        self.psi0 = qt.basis(2, 0)

        # 5. HAMILTONIAN & COLLAPSE OPERATORS
        self.H = 0.5 * self.Omega_R * self.sig_x
        self.c_ops = [np.sqrt(self.gamma_phi - self.gamma_m) * self.sig_z]
        self.sc_ops = [np.sqrt(self.gamma_m) * self.sig_z]

        # 6. SOLVER OPTIONS (Forces QuTiP to keep noise data in memory)
        self.sim_options = {'store_measurement': 'middle'}

    def simulate_single_trace(self):
        """
        Runs a single 4-microsecond simulation of the qubit.
        Returns the 100-step noisy measurement record (V_t) and the final projective outcome (y_T).
        """
        # Run the Stochastic Master Equation solver
        result = qt.smesolve(
            self.H,             
            self.psi0,          
            self.tlist,         
            c_ops=self.c_ops,   
            sc_ops=self.sc_ops, 
            e_ops=[self.sig_z], # We only need to track the Z-axis to find the final state
            ntraj=1,       
            options=self.sim_options
        )

        # Extract the noisy measurement record (V_t) safely across QuTiP versions
        try:
            V_t = result.measurement[0].real 
        except AttributeError:
            try:
                V_t = result.measurements[0][0].real 
            except AttributeError:
                V_t = result.wiener_process[0][0]

        # Flatten the array to ensure it is a clean 1D list of 100 floats
        V_t = np.array(V_t).flatten().real

        # --- SIMULATE THE STRONG PROJECTIVE MEASUREMENT ---
        # result.expect[0] holds the Z-axis trajectory (+1 is state 0, -1 is state 1)
        final_z_expect = result.expect[0][-1] 

        # Convert the final Z-axis expectation into a pure probability of being in state 1 (0.0 to 1.0)
        prob_excited = (1.0 - final_z_expect) / 2.0 

        # Force the state to collapse into a definitive 0 or 1 based on that probability
        y_T = np.random.binomial(1, prob_excited) 

        return V_t, y_T

    def generate_dataset(self, num_traces):
        """
        Loops the simulation to generate a massive dataset for AI training.
        Packages the inputs (X) and targets (Y) into PyTorch Tensors.
        """
        print(f"Generating {num_traces} quantum trajectories. This might take a moment...")
        
        X_data = []
        Y_data = []

        for i in range(num_traces):
            V_t, y_T = self.simulate_single_trace()
            X_data.append(V_t)
            Y_data.append(y_T)
            
            # Simple progress tracker in the console
            if (i + 1) % max(1, (num_traces // 10)) == 0:
                print(f"Progress: {i + 1} / {num_traces} traces generated.")

        # Convert the lists into PyTorch Tensors
        # X shape becomes (batch_size, sequence_length=100, features=1)
        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(-1)
        
        # Y shape becomes (batch_size, 1)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32).unsqueeze(-1)

        print("Dataset generation complete!")
        return X_tensor, Y_tensor

# ==========================================
# TEST RUN THE GENERATOR
# ==========================================
if __name__ == "__main__":
    # Instantiate our new virtual laboratory
    generator = QuantumDataGenerator()
    
    # Generate a tiny batch of 10 traces just to test that it works
    # Note: The paper originally generated 1.5 million traces for training!
    X_train, Y_train = generator.generate_dataset(num_traces=10)
    
    print("\n--- Tensor Shapes ---")
    print(f"X_train shape (The Noisy Sensor Data): {X_train.shape}")
    print(f"Y_train shape (The Final Qubit States): {Y_train.shape}")