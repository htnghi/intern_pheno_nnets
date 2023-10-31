# Basic knowledge of RNN
* RNN are sequence-based learning models, which means they’re used to predict the next event in a sequence of data. They were inspired by the way the brain works as it forms and recalls memories by continuously analyzing previous events in order to decide how to act in the present.

* The key characteristic: ability to maintain a hidden state that captures information from previous time steps. This hidden state allows RNNs to model temporal dependencies in data.

* Applications: including natural language processing (NLP), speech recognition, time series prediction, and more.


https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85
https://towardsdatascience.com/a-brief-introduction-to-recurrent-neural-networks-638f64a61ff4
https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79
https://medium.com/@VersuS_/coding-a-recurrent-neural-network-rnn-from-scratch-using-pytorch-a6c9fc8ed4a7
https://towardsdatascience.com/rnns-from-theory-to-pytorch-f0af30b610e1
https://www.dotlayer.org/en/training-rnn-using-pytorch/
https://hackernoon.com/rnn-or-recurrent-neural-network-for-noobs-a9afbb00e860?source=post_page-----5fb61265d818--------------------------------


* RNN Architecturs:
* **Inputs**: 2 input tensors
    - Input tensor: This tensor should be 1 single step in the sequence. If your total sequence is for example 100 characters of text, then the input will be 1 single character of text
    - Hidden state tensor: This tensor is the hidden state. Remember for the first run of each entire sequence, this tensor will be filled with zeros. Following the example above If you have 10 sequences of 100 characters each (a text of 1000 characters in total) then for each sequence you will generate a hidden state filled with zeros.

* **Weight Matrices**: 3 wieght matrices
    - Input Dense: Dense matrix used to compute inputs (just like feedforward).
    - Hidden Dense: Dense matrix used to compute hidden state input.
    - Output Dense: Dense matrix used to compute the result of `activation(input_dense + hidden_dense)`

* **Outputs**: 2 output tensors
    - New hidden state: New hidden state tensor which is `activation(input_dense + hidden_dense)`. You will use this as input on the next iteration in the sequence.
    - Output: `activation(output_dense)`. This is your prediction vector, which means is like the feedforward output prediction vector

* Different types of RNN:
    - Traditional Recurrent Neural Network (RNN)
    - Long-Short-term-Memory Recurrent Neural Network (LSTM)
    - Gated Recurrent Unit Recurrent Neural Network (GRU)

* Disadvs: vanishing gradient and exploding gradient 
    - Exploding gradient can be fixed with gradient clipping technique 
    - Vanishing gradient limitation was overcome by various networks such as LSTM, GRUs, and residual networks (ResNets).

