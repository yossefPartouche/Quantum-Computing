import torch
import os
import glob
from src.QuantumDataGenerator import QuantumDataGenerator
from src.QuantumLSTM import QuantumLSTM
from src.QuantumEvaluator import QuantumEvaluator
from src.OnlineQuantumTrainer import OnlineQuantumTrainer

def main():
    # --- CONFIGURATION FLAGS ---
    RUN_TRAINING = False      
    MODEL_PATH = 'models/LSTM_QuantumObserver_eps5e-4.pth'
    
    lab = QuantumDataGenerator()
    model = QuantumLSTM()
    
    print("==============================================")
    print(" AI QUANTUM OBSERVER: PIPELINE")
    print("==============================================")

    if RUN_TRAINING:
        print("\n PHASE: ONLINE TRAINING MARATHON")

        trainer = OnlineQuantumTrainer(lab, model)
        trainer.run_infinite_marathon()
        torch.save(model.state_dict(), MODEL_PATH)
    
    else:
        print(f"\n PHASE: EVALUATING CONVERGED MODEL ({MODEL_PATH})")
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file {MODEL_PATH} not found")
            return
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    dashboard = QuantumEvaluator(model)

    print("\nGenerating fresh validation traces...")
    X_val, Y_val = lab.generate_dataset(num_traces=1000)

    with torch.no_grad():
        model_predictions = model(X_val)
    
    dashboard.tomographic_check(model_predictions, Y_val, num_bins=20)
    dashboard.calculate_accuracy(X_val, Y_val)
    dashboard.plot_predictions_vs_reality(X_val, Y_val, num_samples=100)
    dashboard.plot_advanced_metrics(X_val, Y_val, lab)

if __name__ == "__main__":
    main()