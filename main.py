import torch
import os
import glob
from QuantumDataGenerator import QuantumDataGenerator
from QuantumLSTM import QuantumLSTM
from QuantumEvaluator import QuantumEvaluator

def main():
    # --- CONFIGURATION FLAGS ---
    FORCE_REGENERATE = True   # Set to True to start generation
    RUN_TRAINING = True      
    
    # --- DATA PARAMETERS ---
    TOTAL_TRACES = 1500000    # Target from the original research
    BATCH_SIZE = 1000        # Chunks saved to disk to manage RAM
    DATA_DIR = 'quantum_data_batches'
    
    print("==============================================")
    print(" PHASE 1: DATA PREPARATION")
    print("==============================================")
    
    lab = QuantumDataGenerator()
    
    if FORCE_REGENERATE or not os.path.exists(DATA_DIR):
        # This uses all CPU cores via parallel_map to generate 1.5M traces
        lab.generate_massive_dataset(total_traces=TOTAL_TRACES, batch_size=BATCH_SIZE)
    
    # --- LOAD AND COMBINE DATA FOR TRAINING ---
    print("Loading batches from disk...")
    X_files = sorted(glob.glob(f"{DATA_DIR}/X_batch_*.pt"))
    Y_files = sorted(glob.glob(f"{DATA_DIR}/Y_batch_*.pt"))
    
    # For training, we load a subset or use a DataLoader to manage memory
    # Start with a large segment (e.g., 100,000) for high-precision results
    X_train = torch.cat([torch.load(f) for f in X_files[:10]])
    Y_train = torch.cat([torch.load(f) for f in Y_files[:10]])
    
    # Separate validation set
    X_val = torch.load(X_files[-1])
    Y_val = torch.load(Y_files[-1])

    print(f"Dataset ready: {X_train.shape[0]} training traces.")

    print("\n==============================================")
    print(" PHASE 2: TRAINING THE AI")
    print("==============================================")
    brain = QuantumLSTM()
    
    if RUN_TRAINING:
        # 100 Epochs with Learning Rate Sledding and Gradual Dropout
        loss_history = brain.train_model(X_train, Y_train, epochs=100, batch_size=1024)
        torch.save(brain.state_dict(), 'bidirectional_brain.pth')
    else:
        brain.load_state_dict(torch.load('bidirectional_brain.pth'))

    print("\n==============================================")
    print(" PHASE 3: ADVANCED EVALUATION")
    print("==============================================")
    dashboard = QuantumEvaluator(brain)
    dashboard.calculate_accuracy(X_val, Y_val)
    dashboard.plot_advanced_metrics(X_val, Y_val)
    dashboard.plot_predictions_vs_reality(X_val, Y_val, num_samples=10)

if __name__ == "__main__":
    main()