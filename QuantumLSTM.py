import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class QuantumLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        """
        Initializes the neural network architecture.
        """
        super(QuantumLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. The LSTM Layer: 64 artificial neurons to process the 100-step time series
        # batch_first=True ensures it accepts data in the shape (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        
        # 2. Activation Function: Rectified Linear Unit (ReLU)
        self.relu = nn.ReLU()
        
        # 3. Final Output Layer: A fully connected linear layer mapping the 64 neurons down to 1 output
        self.fc = nn.Linear(hidden_size, 1)
        
        # 4. Probability Squasher: Sigmoid function to constrain the final output to a 0.0 - 1.0 probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Dictates how the quantum measurement data flows through the network.
        x represents the input batch of V_t noisy records.
        """
        # Pass the sequence through the LSTM
        # 'out' holds the hidden states for ALL 100 time steps. 
        # '(hn, cn)' holds the final memory state of the sequence.
        out, (hn, cn) = self.lstm(x)
        
        # We only care about the prediction at the very end of the 4 microseconds.
        # out[:, -1, :] extracts the 64-neuron output from the final (100th) time step for every trace in the batch.
        final_time_step_out = out[:, -1, :]
        
        # Apply the ReLU activation function
        activated_out = self.relu(final_time_step_out)
        
        # Pass through the linear layer and squash into a probability
        probability_prediction = self.sigmoid(self.fc(activated_out))
        
        return probability_prediction

    def train_model(self, X_train, Y_train, epochs=50, batch_size=1024, initial_lr=0.001):
        print(f"Initializing Training: Epochs={epochs}, Batch Size={batch_size}")
        
        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=initial_lr)
        
        # Scaling the Learning Rate: Gradually lower it to 0.000001
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_history = [] # FIX: Initialize list to store results

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Implementation of Gradual Dropout would go here if defined in __init__
            
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss) # FIX: Store the average loss
            
            # Step the scheduler to lower learning rate
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        return loss_history # FIX: Return the list so main.py can read its length

# ==========================================
# HOW TO USE THIS CLASS (Example Execution)
# ==========================================
if __name__ == "__main__":
    # Create the network
    model = QuantumLSTM()
    
    # Generate some dummy data representing what QuantumDataGenerator would output
    # Shape: 5000 traces, 100 time steps, 1 feature (voltage)
    dummy_X = torch.randn(5000, 100, 1) 
    
    # Shape: 5000 final measurement outcomes (0 or 1)
    dummy_Y = torch.randint(0, 2, (5000, 1)).float() 
    
    # Train the model
    model.train_model(dummy_X, dummy_Y, epochs=5, batch_size=1024)