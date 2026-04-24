# Research Journey & Methodoology 

### Phase 1: Baseline

1. Goal: Replicate the Berkeley results using a Bidirectional LSTM (64 neurons) and ADAM optimizer.

2. Problem: Limited to 1500 traces due to hardware constraints, resulting in a plateaued accuracy of 72%.

3. Insight: 1.5 million traces are necessary only if the model is trying to "find the signal in the noise." I hypothesized that if the model learns online, it could encounter a more diverse set of "stochastic kicks" in a shorter timeframe.

### Phase 2: The RL-Style Pivot

We shifted from a traditional supervised learning approach to a continuous "Agent" style loop:

Observe: Generate a fresh batch of measurement records.

Evaluate: Use the Bi-LSTM to extract "Quantum Smoothing" (using future noise to inform past states).

Update & Purge: Minimize loss, save the model based on ϵ, and immediately delete the data to free memory.


### Phase 3: Discovering the "Physics Error" ($\epsilon$)

I realized that Loss and Accuracy are misleading in a quantum environment. A model can have low loss by "gambling" on noisy outcomes, but have high Physical Error (poor calibration).

If a qubit is in a superposition stae where the probability of being excited is 0.5 then truth is 50/50. If the model predicts 0.5, the Cross-Entropy loss will be high because the model is "uncertain". Another model could predict the same value with a probability of 0.9 and thus obtain a lower loss, though it may not actually be a true representation of the superposition. Thus, the loss rewards overconfidence whereas physics calibration. Which is why we moved away from the loss.

The Solution: I implemented Tomographic Validation during the training loop.

The Result: We only update "Patience" and save weights when the model demonstrates a better alignment with the Born Rule (the red identity line on the tomography plot), not just when the loss drops.

### Phase 4: Verification via V-Curve

To prove the model wasn't just a "smart filter," we applied a V-Curve Sensitivity Test. By manually shifting the Rabi Frequency (Ω 
R) in the generator, we confirmed that the model's error spikes sharply when the physics deviates from the "Trained Truth." This confirms the RNN has extracted the fundamental physical constants of the system.