**1. How do neural networks work in general? What is a neuron? What is a layer? What kinds of layers exist? What is Deep Learning?**
* Neural networks are a class of machine learning models and inspired by the structure and function of the human brain, translated to the computer. 
* Feedforward neural network: the flow of data through the network in one direction from the input layer through one or more hidden layers to the output layer. At each layer, the data is transformed by weights and biases and passed on to the next layer. There are no connections that loop back to previous layers, or recurrent connections.
* Neuron: also known as a node or a perceptron, is a basic computational unit. It takes one or more input values, processes them, and produces an output.
* Layer: a collection of neurons arranged in a specific way. Layers are connected to each other, forming the network's architecture.    
    - The most common types of layers are:
        - Input Layer: takes the raw input data and passes it to the subsequent layers. Typically, there is one neuron per input feature.
        - Hidden Layers: come between the input and output layers. They process the input data through a series of weighted connections and activations. Deep neural networks have multiple hidden layers.
        - Output Layer: produces the final output of the network. The number of neurons in the output layer depends on the problem type. 
* Deep learning: refers to the training and use of deep neural networks, which are neural networks with multiple hidden layers. Deep learning has gained prominence due to its ability to automatically learn complex patterns from data, making it suitable for a wide range of tasks, such as image and speech recognition, language translation, and more.


**2. How can you express this in equations? What brings in non-linearity and complexity?**
* Each neuron in a layer computes its output by applying a weighted sum to its inputs and passing it through an activation function.
`output = inputs * weights + bias`
* The complexity and non-linearity in neural networks arise from: ...


**3. What is an activation function? What are different options for activation functions and what are their characteristics?**
* Activation function: is a mathematical function that determines the output of a neuron or a node. It introduces non-linearity to the model, allowing neural networks to approximate complex and non-linear patterns in data. 
    - The reason using Activation function: to help the network learn complex patterns in the data.
    - Activation function is used in hidden layers and used in output layers.
* Some options for activation functions and their characteristics:
    - Sigmoid Activation Function: `σ(z) = 1 / (1 + e^(-z))`
        + Outputs values between 0 and 1 -> suitable for binary classification problems where you want to model probabilities.
        + Disadv: Suffers from vanishing gradient problems, which can slow down training in deep networks => Not commonly used in hidden layers of deep networks.
    - Rectified Linear Unit (ReLU) Activation Function: `ReLU(z) = max(0, z)`
        + Outputs the input for positive values and zero for negative values.
        + Advs: solve the vanishing gradient problem and accelerates convergence => speed and efficiency => the most widely used activation function.
        + Disadvs: maybe suffer from the "dying ReLU" problem (neurons that always output zero during training and never update).
    - The Softmax Activation Function: (Output layer for classification)
        + This activation can take non-normalized, or uncalibrated, inputs and produce a normalized distribution of probabilities for each classes. 
        + This distribution returned by the softmax activation function represents ​confidence scores​ for each class and will add up to 1. 
        + The predicted class is associated with the output neuron that returned the largest confidence score.

**4. What is backpropagation? How does it work? Why do you need it?**
* The backpropagation: allows the information from the cost to then flow backward through the network in order to compute the gradient of the loss function with respect to the model's weights and biases. These gradients are then used to update the model's parameters (weights and biases) during the training process, enabling the network to learn from its errors and improve its performance.
    - Example: When we use a feedforward neural network to accept an input x and produce an output yˆ, information flows forward through the network. The input x provides the initial information that then propagates up to the hidden units at each layer and finally produces yˆ.
* How backpropagation works step by step:
    - (1) Forward pass
    - (2) Loss calculation
    - (3) Backpropagation
    - (4) Update weights and biases
    - (5) Iteration
* More details in step (3) Backward pass:
    - Backpropagation is initiated by calculating the gradients of the loss with respect to the output layer's activations. These gradients indicate how much each output node contributes to the overall loss.
    - The gradients are then propagated backward through the network layer by layer. At each layer, the gradients are used to compute the gradients of the loss with respect to the layer's weights and biases.
    - This process is carried out using the **chain rule** of calculus, which allows the gradients to be efficiently calculated for each layer.
* Advs of Backpropagation:
    - Learn from Errors
    - Generalization: Backpropagation helps the network generalize from the training data to make accurate predictions on unseen data.
    - Allows for efficient gradient computation at each layer
    - Programming is quick and simple


**5. What is a loss function? What are examples for classification and regression tasks? Why is it important?**
* Loss functon (also known as a cost function, erroe function or objective function): measures the difference between the predicted values (output of the model) and the actual target values (ground truth) for a given set of input data\
    => used to quantify how well a machine learning or deep learning model is performing on a specific task\
    => The goal of training a model is to minimize this loss function, which means making the predictions as close as possible to the true values.
* The choice of a loss function depends on machine learning task, including classification and regression tasks:
    - Loss Functions for Classification Tasks:
        + Binary Cross-Entropy Loss (Log Loss)
            + Suitable for binary classification problems where there are only two possible classes (e.g., 0 and 1)
        + Categorical Cross-Entropy Loss
            + Used for multi-class classification tasks where there are more than two classes.
            + Measures the dissimilarity between predicted class probabilities and actual class labels.
    - Loss Functions for Regression Tasks:
        + Mean Squared Error (MSE): measures the average squared difference between predicted values and actual target values.
        + Mean Absolute Error (MAE): measures the average absolute difference between predicted values and actual target values.


**6. How do you usually train a neural network? What is an epoch and a batch? What is a batch size and why is it important? What happens if the batch size is large or small? What is gradient descent?**
* Training a neural network involves iteratively updating its parameters (weights and biases) to minimize a loss function, typically using an optimization algorithm like gradient descent.\
The training loop consists of the following steps: 
    - (1) Forward pass
    - (2) Loss calculation
    - (3) Backpropagation
    - (4) Update weights and biases
    - (5) Iteration
* Epochs: An epoch is one complete pass through the entire training dataset. During an epoch, the network processes all training samples, calculates gradients, and updates weights once.
* Batch Size:
    - The batch size determines the number of training examples (data points) used in each forward and backward pass of the training loop. It is a hyperparameter.
    - A batch size of 1 is known as stochastic gradient descent (SGD), while larger batch sizes are often used in practice.
    - Mini-batch sizes (e.g., 32, 64, 128) are common because they provide a balance between computational efficiency and convergence speed.\
Importance of Batch Size:
    - Batch size affects both training speed and model performance.
    - Larger batch sizes can speed up training as they take better advantage of parallelism on modern hardware. However, they may require more memory.
    - Smaller batch sizes can lead to noisier gradient estimates but may help models generalize better.
* Gradient Descent:
    - Gradient descent is an optimization algorithm used to update the model's weights during training.
    - It calculates the gradient of the loss with respect to the model's parameters (weights and biases) and adjusts them in the direction that minimizes the loss.
    - The learning rate hyperparameter controls the step size of weight updates. It's crucial for the convergence of the optimization process.

**7. What is an optimizer in this context? What are examples?**
* The optimizer: an algorithm that adjusts the model's parameters (weights and biases) during the training process to minimize the loss function\
    => determining how much the parameters should be updated at each iteration (step) of training
* Some common optimizers:
    - Gradient Descent (GD)
    - Momentum
    - Adagrd
    - RMSprop
    - Adam

**8. What is a learning rate? Why is the learning rate important? What happens if you choose a large or a small learning rate? What is learning rate scheduling, how does it work and why could it be senseful?**
* **The learning rate**: a hyperparameter that controls the step size or the size of weight updates made during each iteration of training. 
    - It is a small positive value typically set before training begins.
    - Learning rate determines how quickly or slowly a neural network converges during training.
* Importance of Learning Rate:
    - The learning rate is a crucial hyperparameter because it influences the convergence and stability of training.
    - Proper choice of the learning rate is essential to ensure that the optimization process converges to an optimal solution without diverging or getting stuck in local minima.
* Effects of Learning Rate Choice:
    - Large Learning Rate: If the learning rate is too large, weight updates can overshoot the optimal values, causing the optimization process to diverge, and the loss may increase instead of decreasing. Training may become unstable.
    - Small Learning Rate: If the learning rate is too small, weight updates are too tiny, and the training process may become extremely slow. It may also get stuck in local minima without converging to a good solution.
* Learning rate scheduling is a technique where the learning rate is adjusted during training. The learning rate is often decreased (annealed) over time.\
    => It can be beneficial for improving convergence and fine-tuning hyperparameters.\
    => Techniques like step decay, exponential decay, and cyclical learning rates are commonly used forms of learning rate scheduling.
    * Advantages of Learning Rate Scheduling:
        - Faster Convergence: Learning rate scheduling can speed up convergence by allowing the model to make larger updates in the early stages of training when gradients are large and gradually reducing the learning rate as training progresses.
        - Stability: It helps stabilize training by preventing abrupt weight updates that can lead to divergence.
        - Escape Local Minima: Adaptive learning rate schedules can help the model escape local minima by allowing it to explore the loss landscape more effectively.

**9. What is a training loss and what is a validation loss? How should these compare to each other during a training process? What can you determine depending on a certain relationship between the training and validation loss?**

| Training loss                                                                                                                                                        | Validation loss                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| a measure of how well the model is performing on the training data during each training iteration (epoch)                                                            | a measures of how well the model generalizes to unseen data                                                  |
| quantifies the difference between the model's predictions and the actual target values for the training dataset                                                      | quantifies the difference between the model's predictions and the actual target values for a separate dataset called the validation dataset. |
| The goal is to minimize this loss during training by adjusting the model's parameters (weights and biases) using an optimization algorithm (e.g., gradient descent). | The validation dataset is not used for training, it is set aside specifically for evaluation purposes.                                       |


=> The relationship between the training loss and the validation loss provides insights into the model's performance and its ability to generalize to unseen data.
* Comparison between Training Loss and Validation Loss during a training process:
    - Initially, both the training loss and validation loss should decrease because the model learns from the training data\
    => This indicates that the model is making progress and learning the underlying patterns in the data.
    - Continuing training, the training loss keep decreasing, while the validation loss may reach a minimum value and then start to increase or plateau.
    - Some scenarios:
        + <u>Ideal</u>: **Training Loss < Validation Loss (Both are low)**\
                Explain: The model is learning from the training data and generalizing effectively to the validation data. There is no significant overfitting, and the model is likely to perform well on unseen data.\
                Another case: **Training Loss ≈ Validation Loss (Both are low)**\
                               Explain: There may be some overfitting a little bit. But the model is still good.
        + <u>Underfitting</u>: **Training Loss : too high (Validation loss: also too high)**\
                        Explain: The model is not learning from the training data or is too simple to capture the underlying patterns in the data
        + <u>Overfitting</u>: **Training Loss << Validation Loss**\
                      Explain: he model is fitting the training data too closely and is not generalizing well to new data.


**10. Why is overfitting important? What are different options to prevent overfitting in a neural network? What is dropout and how does it work? What is early stopping and how does it work? What is L1-regularization and how does it work in the context of neural networks?**
* Overfitting: a common problem in machine learning and neural networks that occurs when a model learns the training data too well, capturing noise and specific examples in the data rather than the underlying patterns.
    - Why overfitting is important? => lead to poor generalization, meaning the model performs well on the training data but poorly on unseen data. 
* Techniques to prevent overfitting:
    - (1) Reducing Model Complexity. e.g. fewer neurons in hidden layers or fewer layer.
    - (2) Validation Data
    - (3) Regularization
    - (4) Dropout layer
    - (5) Early Stopping
* **Dropout**: randomly "dropping out" a fraction of neurons (units) during each training iteration.
    + Dropout is typically applied to the input and hidden layers during training but is turned off during inference (testing) when making predictions.
* **Early stopping**: a technique that monitors the validation loss during training and stops training when the validation loss begins to increase (indicating overfitting).
* **L1 and L2 Regularization**: Add penalty number to the loss function to penalize the model for large w and b.
    - L1: encourages sparsity in weights, effectively setting some weights to exactly zero => help with feature selection and simplifying the model
    - L2: encourages the model to have smaller weights overall => effectively discourages the model from relying too heavily on any single weight => reducing the risk of overfitting.