import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class QuantumEvaluator:
    def __init__(self, model):
        """
        Initializes the dashboard with your trained LSTM or Transformer[cite: 598].
        """
        self.model = model

    def plot_training_loss(self, loss_history):
        """
        Visualizes the learning curve. You should see it plummet from 0.693 
        as the weights tune to the physics[cite: 599, 708, 800].
        """
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='AI Learning Curve', color='blue', linewidth=2)
        plt.axhline(y=0.693, color='red', linestyle='--', label='Random Guessing (0.693)')
        plt.title("Phase 2: Training Loss (Cross-Entropy)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def calculate_accuracy(self, X_val, Y_val):
        """
        Calculates a high-level percentage of correct binary outcomes[cite: 602, 780].
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val)
            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == Y_val).sum().item()
            accuracy = (correct / Y_val.size(0)) * 100
            print(f"\n--- AI VALIDATION RESULTS ---")
            print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def plot_predictions_vs_reality(self, X_val, Y_val, num_samples=10):
        """
        A quick visual check showing what the AI guessed vs what actually happened.
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
            print(f"Trace {i+1}: AI Guessed {prob:.2f} (Class {guess}) | Actual: {actual} {status}")

    def plot_advanced_metrics(self, X_val, Y_val):
        """
        Generates the Confusion Matrix and ROC Curve to reveal AI bias 
        and statistical understanding[cite: 767, 769, 792].
        """
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_val).detach().numpy()
            preds = np.round(probs)
            actuals = Y_val.detach().numpy()

        # 1. Confusion Matrix: Shows if AI struggles with Ground vs Excited 
        cm = confusion_matrix(actuals, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix: Ground vs. Excited")
        plt.ylabel("Actual State")
        plt.xlabel("AI Predicted State")
        plt.show()

        # 2. ROC Curve: Proves model understands quantum uncertainty 
        fpr, tpr, _ = roc_curve(actuals, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title("Receiver Operating Characteristic")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()