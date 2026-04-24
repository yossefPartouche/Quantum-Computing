import torch
import itertools
import torch.optim as optim
from src.QuantumDataGenerator import QuantumDataGenerator
from src.QuantumLSTM import QuantumLSTM
from src.QuantumEvaluator import QuantumEvaluator
import numpy as np

class OnlineQuantumTrainer: 
    def __init__(self, lab, model, val_size=5000):
        self.lab = lab
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()
        self.X_val, self.Y_val = self.lab.generate_dataset(num_traces = val_size)
    
    def calculate_relative_error(self, num_bins=20):
        """Quantifies accuracy using the paper's epsilon formula"""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_val).cpu().numpy().flatten()
            actuals = self.Y_val.cpu().numpy().flatten()

        bins = np.linspace(0, 1, num_bins + 1)
        epsilon = 0
        total_n = len(preds)

        for i in range(num_bins):
            indices = np.where((preds >= bins[i]) & (preds < bins[i+1]))[0]
            if len(indices) > 0:
                n_p = len(indices)
                p_center = (bins[i] + bins[i+1]) / 2
                avg_y_sp = np.mean(actuals[indices])
                # Formula from the research paper 
                epsilon += (n_p / total_n) * (avg_y_sp - p_center)**2
        return epsilon

    
    def train_step(self, num_traces, dropout_p):
        # First Generate the Batch
        X, Y = self.lab.generate_dataset(num_traces=num_traces)

        self.model.train()
        self.model.dropout.p = dropout_p
        self.optimizer.zero_grad()

        predictions = self.model(X)
        loss = self.criterion(predictions, Y)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def run_infinite_marathon(self, batch_size = 1000, check_every=10, patience=20):
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9965)
        best_epsilon = float('inf')
        no_improvement_count = 0

        print(f"Starting infinite Online Mining: Press CMD +C to stop manually!")

        for step in itertools.count(start=1):
            dropout_p = 0.3 * (0.9995 ** step)

            loss = self.train_step(batch_size, dropout_p)
            scheduler.step()
            
            if step % check_every == 0:
                current_epsilon = self.calculate_relative_error()

                print(f"Batch {step} | Loss: {loss:.6f} | Physics Error (ε): {current_epsilon:.6f}")

                if current_epsilon < best_epsilon:
                        best_epsilon = current_epsilon
                        no_improvement_count = 0
                        torch.save(self.model.state_dict(), 'best_physics_model.pth')
                else:
                    no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"Physics Converged! ε stopped improving after {step * batch_size} traces.")
                break






        


