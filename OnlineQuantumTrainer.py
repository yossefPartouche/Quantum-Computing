import torch
import torch.optim as optim
from QuantumDataGenerator import QuantumDataGenerator
from QuantumLSTM import QuantumLSTM
from QuantumEvaluator import QuantumEvaluator

class OnlineQuantumTrainer: 
    def __init__(self, lab, model):
        self.lab = lab
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()
    
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
    
    def run_marathon(self, total_traces=5000000, batch_size = 100, patience=50):
        num_steps = total_traces // batch_size
        
        gamma = (1e-6/ 0.001)**(1 / num_steps)
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        best_loss = float('inf')
        no_improvement_count = 0

        print(f"Starting Online Mining: Target {total_traces} traces.")

        for step in range(num_steps):
            dropout_p = 0.3 * (1 - (step / num_steps))

            loss = self.train_step(batch_size, dropout_p)
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                no_improvement_count = 0
                torch.save(self.model.state_dict(), 'model_02.pth')
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"Stopping Early: Model converged at {step*batch_size} traces.")
                break
                
            if step % 10 == 0:
                print(f"Batch {step}/{num_steps} | Loss: {loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")

def main():
    lab = QuantumDataGenerator()
    model = QuantumLSTM()

    trainer = OnlineQuantumTrainer(lab, model)
    trainer.run_marathon(total_traces=5000000, batch_size = 100)
    
    model.load_state_dict(torch.load('model_02.pth'))
    dashboard = QuantumEvaluator(model)
    X_val, Y_val = lab.generate_dataset(num_traces=100)
    model.eval()
    with torch.no_grad():
        model_predictions = model(X_val)

    dashboard.tomographic_check(model_predictions, Y_val, num_bins=20)
    dashboard.calculate_accuracy(X_val, Y_val)
    dashboard.plot_advanced_metrics(X_val, Y_val)
    dashboard.plot_predictions_vs_reality(X_val, Y_val, num_samples=100)

if __name__ == "__main__":
    main()




        


