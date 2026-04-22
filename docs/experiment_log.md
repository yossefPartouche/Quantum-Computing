I'm following the implementation of a paper, which used RNN (LSTM) to reconstruct Quantum Dynamics of a qubit. 
The researches generated 1.5M traces and was able to obtain 99%+ Accuracy, I do not have the computational power to create such number of traces and at best was able to generate 1500 traces and reach 72% accuracy, whilst using a Bidirectional LSTM (64 neurons) and ADAM with LR sleddding. 
This led to two conclusions: (i) for 1.5M was only for a single qubit what about a multiple or traces of more than two states? (ii) This doesn't necessarily imply that the model is learning the universal physics.
So we now try using RL methods to save both computational expense and always have the model learn on single finite set of traces continously. 
I want to try implement and RL-style method: 
    - Generate fresh batch of measurement records V_t
    - Evaluate (Agent): Bidirectional LSTM to extract "Quantum Smoothing" information
    - Update & Purge: Minimize the cross-entropy loss relative to the projective outcome and overwrite batch
I added the tomography test since the physical environment is inherently probabilistic, thus the loss and accuracy don't exactly reflect the performance of the model. 
