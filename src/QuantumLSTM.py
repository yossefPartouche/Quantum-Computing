import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class QuantumLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(QuantumLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. BIDIRECTIONAL LSTM LAYER
        # We add 'bidirectional=True'. This creates two hidden layers (Forward & Backward).
        # The output size will now be hidden_size * 2 (64 * 2 = 128 neurons).
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=True)
        
        # 2. GRADUAL DROPOUT LAYER
        # We initialize it at 0.3 (30%) as per the paper's first epoch strategy.
        self.dropout = nn.Dropout(p=0.3)
        
        self.relu = nn.ReLU()
        
        # 3. FINAL OUTPUT LAYER
        # Since the LSTM is bidirectional, we must accept 128 inputs (64 from each direction).
        self.fc = nn.Linear(hidden_size * 2, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out shape: (batch, sequence, hidden_size * 2)
        out, (hn, cn) = self.lstm(x)
        
        # For Quantum Smoothing, we take the final hidden states from BOTH directions.
        # We use the output from the final time step which contains the concatenated 
        # information from the forward and backward passes.
        final_out = out[:, -1, :]
        
        # Apply the current level of Dropout
        dropout_out = self.dropout(final_out)
        
        # Apply ReLU and squash to probability
        activated_out = self.relu(dropout_out)
        return self.sigmoid(self.fc(activated_out))

    def train_model(self, X_train, Y_train, epochs=100, batch_size=1024, initial_lr=0.001):
        """
        Implements surgical Hyperparameter Optimization (HPO):
        1. Learning Rate Sledding (Exponential Decay)
        2. Gradual Dropout Reduction (0.3 to 0.0)
        """
        print(f"Initializing Bidirectional Training: Epochs={epochs}, Batch Size={batch_size}")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=initial_lr)
        
        # HPO 1: LEARNING RATE SLEDDING
        # We want to reach 0.000001 over the number of epochs. 
        # We calculate the decay rate (gamma) mathematically to hit that target precisely.
        gamma = (0.000001 / initial_lr) ** (1 / epochs)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_history = []

        for epoch in range(epochs):
            self.train()
            
            # HPO 2: GRADUAL DROPOUT REDUCTION
            # Linearly reduce dropout from 0.3 at Epoch 0 to 0.0 at the final Epoch.
            current_dropout = 0.3 * (1 - (epoch / epochs))
            self.dropout.p = current_dropout
            
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            # Update Learning Rate
            scheduler.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.7f} | Dropout: {self.dropout.p:.3f}")
        
        return loss_history