Hello, this is my webpage

#### Machine Learning Bounds

I gather all these information from ChatGPT. It is a fast way to search for relevant information as it is difficult to find the specific webpage on each topic. You should try to derive the bounds. Although you cannot derive new bounds, deriving old bounds help you understand the techniques and algorithms better.

### batch size
<img src="Screenshot 2025-08-09 at 3.38.04 PM.png" alt="alt text" width="400">

Deep learning does stochastic gradient descent. So the gradient is modeled as a determinstic gradient with Gaussian noise added. The variance of the gradient decreases with increase in batch size.

<img src="Screenshot 2025-08-09 at 3.41.55 PM.png" alt="alt text" width="400">

This formula computes the generalization gap of stochastic gradient descent. nt is the learning rate, t is the number of steps, n is the dataset size, and lastly B is the batch size. Increase in batch size will increase learning accuracy assume that there is no overfitting. 
### gradient descent

<img src="Screenshot 2025-08-09 at 3.46.12 PM.png" alt="alt text" width="400">

The convergence rate is O(1/t) in deterministic gradient descent. t is the number of gradient descent steps. Basically it means that the loss function will descrease at rate 1/t.

<img src="Screenshot 2025-08-09 at 3.49.53 PM.png" alt="alt text" width="400">

If the loss function is not skewed, e.g. in quadratic loss function, the condition number (largest eigenvalue divided by lowest eigenvalue is small), the loss function will look like a circular bowl with no zig-zag in gradient descent, so it will converge in exponential rate.

<img src="Screenshot 2025-08-09 at 3.52.56 PM.png" alt="alt text" width="400">

For stochastic gradient descent, the decrease rate of loss function is 1/sqrt(t), which is much slower than ordinary gradient descent.

<img src="Screenshot 2025-08-09 at 3.56.23 PM.png" alt="alt text" width="400">

The error bound of stochastic gradient descent can be summarized by 2 terms, one is the deterministic rate 1/t and another is the non-deterministic rate determined by the batch size B.

## stochastic, newton, fixed and decreasing learning rate, conjugate accelerated nesterov gd, stability, sensitivity, differential equations bounds
### vc dim
Vapnik–Chervonenkis dimension is a measure of the capacity of a machine learning model or algorithm. How much information can it learns. Higer VC dim is usually better but it may overfit and learn the noise in the data. We can further fix this by using regularization, which I will talk more in the later part. Some models such as K nearest neighbors classifier has infinite VC dimension. The VC dimension will keep on increasing as the number of data points increase. Non parametric probbability distribution estimation is another example.

## decision tree, svm, quadratic kernels,
<img src="Screenshot 2025-08-09 at 3.15.44 PM.png" alt="alt text" width="400">

The VC dimension of linear SVM is d+1. Because the hyperplane can shatter the input space into D + 1 possible regions. This is the same for perceptron (1 layer) in deep learning. 

<img src="Screenshot 2025-08-09 at 3.19.12 PM.png" alt="alt text" width="400">

For soft margin SVM, the VC dimension increases with slack variable. The slack variable controls the regularization of learning. The higher the slack variable, the lesser the regularization. Usually with high dimension input, we add regularization to avoid overfitting. In the case of deep learning, usually we create a large network to overfit the data, then we add regularization to increase the validation accuracy on the test dataset. The variable inside the f(.) function is the slack variable.

<img src="Screenshot 2025-08-09 at 2.54.38 PM.png" alt="alt text" width="400">

This is the VC dimension of decision tree. Basically it means that decision tree will shatter the input space into 2 to the power of D subspace as represented by the leaves, with D different classes at the leaves. 

## multilayer perceptron, ensembles max and mean outputs, compare similarity
## transformer, attention
## group conv, alexnet double branches bound
### chevesky bound
## variance reduction
### convexity
## super, l smooth learning rate, cross entropy l smooth factor, quadratic with large condition number pl condition
### sparsity learning
### svm number of training samples
### l2 regularization bound
### pca, lda, ica bounds
### geometric complexity entropy bound
### matrix svd bounds
### genetic algo, annealing, ant bounds
### markov chain bounds
