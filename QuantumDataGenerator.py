import qutip as qt
import numpy as np
import torch
import os

class QuantumDataGenerator:
    def __init__(self):
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

        # 4. INITIAL STATE
        self.psi0 = qt.basis(2, 0)

        # 5. HAMILTONIAN & COLLAPSE OPERATORS
        self.H = 0.5 * self.Omega_R * self.sig_x
        self.c_ops = [np.sqrt(self.gamma_phi - self.gamma_m) * self.sig_z]
        self.sc_ops = [np.sqrt(self.gamma_m) * self.sig_z]

        # 6. UPDATED SOLVER OPTIONS
        self.sim_options = {
            'store_measurement': 'middle',
            'keep_runs_results': True,
            'map' : 'loky',
            'num_cpus' : 8

        }

    def generate_dataset(self, num_traces):
        """
        Generates a batch of data using parallel processing.
        """
        # In QuTiP 5.x, map_func is passed directly or handled by the solver
        # We also pass ntraj to generate the batch in parallel
        result = qt.smesolve(
            self.H,             
            self.psi0,          
            self.tlist,         
            c_ops=self.c_ops,   
            sc_ops=self.sc_ops, 
            e_ops=[self.sig_z],
            ntraj=num_traces,
            options=self.sim_options,
            # Parallelization is now often handled here or automatically
            # If your system supports it, QuTiP 5 uses parallel_map by default for ntraj > 1
        )

        X_data = []
        Y_data = []

        # Process the results
        for i in range(num_traces):
            try:
                # Syntax for extracting measurement noise can vary by sub-version
                raw_noise = result.measurement[i][0].real
            except (AttributeError, IndexError):
                try:
                    raw_noise = result.measurements[i][0].real
                except (AttributeError, IndexError):
                    raw_noise = result.wiener_process[i][0]

            V_t = np.array(raw_noise).flatten().real
            X_data.append(V_t)

            # Simulate the final strong measurement
            final_z_expect = result.expect[0][i][-1] 
            prob_excited = (1.0 - final_z_expect) / 2.0 
            y_T = np.random.binomial(1, prob_excited) 
            Y_data.append(y_T)

        X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(-1)
        Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32).unsqueeze(-1)

        return X_tensor, Y_tensor

    def generate_massive_dataset(self, total_traces=1500000, batch_size=10000):
        # Create directory if it doesn't exist
        DATA_DIR = 'quantum_data_batches'
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        num_batches = total_traces // batch_size
        print(f"Starting massive generation: {total_traces} traces in {num_batches} batches.")

        for b in range(num_batches):
            print(f"Processing Batch {b+1}/{num_batches}...")
            X_batch, Y_batch = self.generate_dataset(num_traces=batch_size)
            
            torch.save(X_batch, f'{DATA_DIR}/X_batch_{b}.pt')
            torch.save(Y_batch, f'{DATA_DIR}/Y_batch_{b}.pt')
            
        print("Massive dataset generation complete!")