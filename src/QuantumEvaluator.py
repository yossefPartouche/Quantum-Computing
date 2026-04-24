import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class QuantumEvaluator:
    def __init__(self, model):
        """
        Initializes the dashboard with your trained LSTM or Transformer.
        """
        self.model = model

    def calculate_accuracy(self, X_val, Y_val):
        """
        Calculates a high-level percentage of correct binary outcomes.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val)
            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == Y_val).sum().item()
            accuracy = (correct / Y_val.size(0)) * 100
            print(f"\n--- Model VALIDATION RESULTS ---")
            print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def plot_predictions_vs_reality(self, X_val, Y_val, num_samples=10):
        """
        A quick visual check showing what the mmodel guessed vs what actually happened.
        """
        self.model.eval()
        with torch.no_grad():
            # Get the probability predictions for the first few samples
            predictions = self.model(X_val[:num_samples])
            
        print("\n--- SAMPLE PREDICTIONS ---")
        for i in range(num_samples):
            prob = predictions[i].item()
            guess = int(round(prob)) # 1 if prob >= 0.5, else 0
            actual = int(Y_val[i].item())
            
            status = "✅" if guess == actual else "❌"
            print(f"Trace {i+1}: Model Guessed {prob:.2f} (Class {guess}) | Actual: {actual} {status}")
    
    def tomographic_check(self, model_predictions, actual_outcomes, num_bins=20, plot=True):
        if torch.is_tensor(model_predictions):
            model_predictions = model_predictions.detach().cpu().numpy().flatten()
        if torch.is_tensor(actual_outcomes):
            actual_outcomes = actual_outcomes.detach().cpu().numpy().flatten()

        n_bins = int(num_bins)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        true_averages = []
        valid_bin_centers = []

        for i in range(n_bins):
            indices = np.where((model_predictions >= bins[i]) & (model_predictions < bins[i+1]))[0]
            if len(indices) > 0:
                avg_reality = np.mean(actual_outcomes[indices])
                true_averages.append(avg_reality)
                valid_bin_centers.append(bin_centers[i])

        # Calculate epsilon (Physics Error) as the Mean Squared Error from identity
        epsilon = np.mean((np.array(valid_bin_centers) - np.array(true_averages))**2)

        if plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(valid_bin_centers, true_averages, color='blue', label=f'RNN (ε={epsilon:.6f})')
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect Physics')
            plt.xlabel('Predicted Probability (p)')
            plt.ylabel('Averaged Outcome ⟨y⟩_Sp')
            plt.title('Tomographic Validation')
            plt.legend()
            plt.show()

        return epsilon
    
    def run_v_curve_test(self, generator, param_name='Omega_R'):
        scales = np.linspace(0.8, 1.2, 9) # Test from 80% to 120% of true value
        errors = []
    
        print(f"Starting V-Curve Test for {param_name}...")
        original_value = getattr(generator, param_name)

        for scale in scales: 
           # Temporarily modify the generator's physics
            setattr(generator, param_name, original_value * scale)

            # Generate 'Alien' data with the wrong physics
            X_alien, Y_alien = generator.generate_dataset(1000)

            # Evaluate using model
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_alien)
            
            epsilon = self.tomographic_check(preds, Y_alien, plot=False)
            errors.append(epsilon)

            # Reset physics
            setattr(generator, param_name, original_value)
            print(f"Scale {scale:.2f} (Value: {original_value*scale:.2f}) | Physics Error: {epsilon:.6f}")

        return scales, errors

    
    def plot_advanced_metrics(self, X_val, Y_val, generator):
        """
        Generates the Confusion Matrix and ROC Curve to reveal AI bias 
        and statistical understanding
        """
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_val).detach().numpy()
            preds = np.round(probs)
            actuals = Y_val.detach().numpy()

        # 1. Confusion Matrix
        cm = confusion_matrix(actuals, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix: Ground vs. Excited")
        plt.show()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(actuals, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

        scales, errors = self.run_v_curve_test(generator, param_name='Omega_R')
        plt.figure(figsize=(8,5))
        plt.plot(scales, errors, 'o-', linewidth=2)
        plt.axvline(1.0, color='red', linestyle='--', label='Trained Value (1.0x)')
        plt.title("Rabi Frequency Sensitivity (V-Curve)")
        plt.xlabel("Scale Factor (Actual / Trained)")
        plt.ylabel("Physics Error (epsilon)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()