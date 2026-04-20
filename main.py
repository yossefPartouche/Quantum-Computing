import torch
import os
from QuantumDataGenerator import QuantumDataGenerator
from QuantumLSTM import QuantumLSTM
from QuantumEvaluator import QuantumEvaluator

def main():
    # --- CONFIGURATION FLAGS ---
    # Set to True to force new data generation even if files exist
    FORCE_REGENERATE = False 
    # Set to True to run the training phase
    RUN_TRAINING = True      
    
    # --- DATA PARAMETERS ---
    TRAIN_SIZE = 5000
    VAL_SIZE = 1000
    DATA_FILES = ['X_train.pt', 'Y_train.pt', 'X_val.pt', 'Y_val.pt']
    
    print("==============================================")
    print(" PHASE 1: DATA PREPARATION")
    print("==============================================")
    
    # Logic: Skip generation if files exist AND FORCE_REGENERATE is False
    if not FORCE_REGENERATE and all(os.path.exists(f) for f in DATA_FILES):
        print("Existing datasets found on disk. Loading...")
        X_train, Y_train = torch.load('X_train.pt'), torch.load('Y_train.pt')
        X_val, Y_val = torch.load('X_val.pt'), torch.load('Y_val.pt')
    else:
        print("Starting virtual quantum lab for data generation...")
        lab = QuantumDataGenerator()
        X_train, Y_train = lab.generate_dataset(num_traces=TRAIN_SIZE)
        X_val, Y_val = lab.generate_dataset(num_traces=VAL_SIZE)
        
        # Save tensors for future reuse
        torch.save(X_train, 'X_train.pt')
        torch.save(Y_train, 'Y_train.pt')
        torch.save(X_val, 'X_val.pt')
        torch.save(Y_val, 'Y_val.pt')
        print("Quantum datasets saved to disk.")

    print("\n==============================================")
    print(" PHASE 2: TRAINING THE AI")
    print("==============================================")
    brain = QuantumLSTM()
    loss_history = []

    if RUN_TRAINING:
        # Run training and store the history for visualization
        loss_history = brain.train_model(X_train, Y_train, epochs=40, batch_size=1024)
        # CRITICAL: Save the weights after training so you can skip training next time
        torch.save(brain.state_dict(), 'model_weights.pth')
        print("Training complete. Weights saved to 'model_weights.pth'.")
    else:
        # Load pre-trained weights if skipping training
        if os.path.exists('model_weights.pth'):
            print("Skipping training. Loading pre-trained weights...")
            brain.load_state_dict(torch.load('model_weights.pth'))
        else:
            print("WARNING: No saved weights found! Model will use random neurons.")

    print("\n==============================================")
    print(" PHASE 3: ADVANCED EVALUATION")
    print("==============================================")
    dashboard = QuantumEvaluator(brain)
    
    dashboard.calculate_accuracy(X_val, Y_val)
    
    # Only plot loss if training was actually performed
    if len(loss_history) > 0:
        dashboard.plot_training_loss(loss_history)
        
    dashboard.plot_advanced_metrics(X_val, Y_val)
    dashboard.plot_predictions_vs_reality(X_val, Y_val, num_samples=10)

if __name__ == "__main__":
    main()