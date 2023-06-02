# ML-Expert  [`link`](https://www.algoexpert.io/machine-learning/product)
In this repository I will post all the knowledge that I got from the ML Expert course

## A/A Test 

An A/B test in which the experiences being tested are identical to one another. This is done in an effort to determine statistical validity of the A/B tool, the metric being monitored, and the analysis process being used. 

## A/B Testing 

The process of providing two or more different experiences across two or more subgroups of a population. The goal is to measure the change in behavior of the subgroups upon receiving the respective experiences. 

## Accuracy 

The number of true positives plus the number of true negatives divided by the total number of examples. 

## Activation Function 

The function used on the output of a neuron. These activations can be linear or nonlinear. If they're nonlinear, they can be symmetric or asymmetric. 

## Adagrad 

An optimizer used to update the weights of a neural network in which different learning rates are applied to different weights. 

## Adam 

A common gradient descent optimizer that takes advantage of momentum and adaptive learning rates. 

## Agglomerative Clustering 

A clustering algorithm that builds a hierarchy of subclusters that gradually group into a single cluster. Some techniques for measuring distances between clusters are single-linkage and complete-linkage methods. 

## Amazon Kinesis 

An AWS product that provides a way to handle real-time data streaming. 

## Apache Airflow 

A workflow management system that provides users a way to author, schedule, and execute software. 

## Apache Kafka 

An open-source software platform which provides a way to handle real-time data streaming. 

## Apache Spark 

A software interface which provides implicit parallelism and fault tolerance across a cluster of machines. 

## Apache YARN 

A software product responsible for managing compute resources across a cluster of machines. 

## Arithmetic Mean 

The average of a series of numbers. 

## Automated Machine Learning (Auto-ML) 

A strategy which automates the process of applying features and labels to a machine learning model. 

## Availability Zone (AZ) 

Typically a single data center within a region which has two or more data centers. The term "multi-AZ" implies that an application or software resource is present across more than one AZ within a region. This strategy allows the software resource to continue operating even if a data center becomes unavailable. 

## Avro 

A row-oriented data serializer provided by Apache. 

## Backpropagation 

The use of the derivative chain rule along with dynamic programming to determine the gradients of the loss function in neural networks. 

## Backpropagation Through Time 

Backpropagation within an RNN in which an additional partial derivative term is calculated per time step that the input required. 

## Bagging 

Also bootstrap aggregation, a sampling technique which selects subsets of examples and/or features to train an ensemble of models. This generally reduces the variance error. 

## Batch Normalization 

The process of normalizing values by means of re-centering (subtracting the mean) or re-scaling (dividing by the standard deviation) before they go to subsequent layers in a neural network. The goal is to accelerate the learning of deep neural networks by decreasing the number of epochs required for the loss function to converge. 

## Bernoulli Distribution 

A distribution which evaluates a particular outcome as binary. In the Bernoulli Naive Bayes classifier case, a word was either in a message or not in a message, which is binary representation. 

## Beta Distribution 

This distribution is used to model percentages and proportions such as click-through probabilities. 

## Bias-variance Tradeoff 

A pursuit of balance between the errors produced by the bias and the variance of a particular model. The higher the bias, or assumptions injected into the model, typically the more bias error is produced. This generally results in a model which underfits the training examples, which means the assumptions limit the model's ability to properly learn. 

In contrast, the fewer assumptions are injected into the model, the more variance is allowed in the model parameters, which means the model may learn from the noise in the training examples. This generally results in a model which overfits the training examples, which means the lack of assumptions limit the model's ability to properly learn. 

The errors from the bias and variance, combined with the irreducible error, make up the overall model error. Both overfitting and underfitting can inhibit the model's ability to generalize or predict the labels of the test examples. 

## Binary Classification 

A supervised learning task in which there are two possible outputs. 

## Bisected K-means 

Using k-means with k equal to 2 for subsets of the examples in an effort to reduce the chance of converging to a local optima. 

## Boosting 

An ensemble technique which trains many weak learners sequentially, with each subsequent weak learner being trained on the previous weak learner's error. This generally reduces the bias error. 

## Candidate Generator 

A system which outputs the candidates to be ranked. This is sometimes referred to as retrieving the ‘top-k' 

## Cell State 

Used within a Long short-term memory (LSTM) acting as an additional hidden state. 

## Central Processing Unit (CPU) 

A general purpose compute resource responsible for executing computer programs. 

## Centroid 

The location of the center of a cluster in n-dimensions. 

## Change Data Capture 

The process of recording changes in the data within a database system. For instance, if a user cancels their Netflix subscription, then the row in some table will change to indicate that they're no longer a subscriber. 

The change in this row can be recorded and referenced later for analysis or audit purposes. 

## Channels 

The number of stacked images, typically representing Red Green Blue pixels, or the number of feature maps produced by a convolutional layer. 

## Classification and Regression Tree 

Also CART, is an algorithm for constructing an approximate optimal decision tree for given examples. 

## Clickstream 

An ordered series of interactions that users have with some interface. In the traditional sense, this can be literal clicks of a mouse on a desktop browser. Interactions can also come from touchscreens and conversational user interfaces. 

## Closed-Form Solution 

For our case, this is what ordinary least squares provides for linear regression. It's a formula which solves an equation. 

## Cluster 

A consolidated group of points. 

## Coefficient 

Another name for a parameter in the regression model. 

## Cold-start 

A challenge with recommendation systems in which users or items are new and that there is limited or no information in terms of the user-item matrix. This can make personalized recommendations difficult or impossible. 

## Collaborative Filtering 

A recommendation technique which uses many user's and item's data typically in the form of a user-item matrix. 

## Collinearity 

When one or more (multicollinearity) independent variables are not actually independent. 

## Concurrency 

When two or computer programs share a single processor. 

## Confidence Interval 
 
The possible range of an unknown value. Often comes with some degree of probability e.g. 95% confidence interval. 

## Confusion Matrix 

In the binary case, a 2x2 matrix indicating the number of true positives, true negatives, false positives and false negatives. 

## Constrained Optimization 

Opposed to an unconstrained optimization problem in which we can use gradient descent, optimization problems with constraints usually require other optimization methods. If there are equality constraints, Lagrange multipliers can often be used. If there are inequality constraints, then linear or quadratic programming can often be used. 

## Content Filtering 

A recommendation technique which takes into account a single user's features and many items' features. 

## Convex Function 
Function with one optima.

## Convolutional Layer

Similar to a fully connected neural layer; however, each neuron is only connected to a subset of neurons in the subsequent layer with respect to a receptive field.

## Cookie

A small piece of data stored by a browser which indicates stateful information for a particular website. For instance, a cookie can be stored in your browser after you log in to a website to indicate that you are logged in. This will stop subsequent pages from asking you to log in again.

## Correlation

The relationship between a dependent and independent variable.

## Cosine Similarity

A similarity metric which is equal to the cosine of the angle between two inputs in some embedding space.

## Cross-Entropy Loss

A loss function which is used in classification tasks. It's technically the entropy of the true labels plus the KL-divergence of the predicted and true labels. Minimizing the cross-entropy minimizes the difference between the true and predicted label distributions.

## Data Governance

The method of managing, using, and protecting an organization's data.

## Data Ingestion

A subsystem responsible for processing incoming data. Typically, the data comes from clickstreams, change data captures, or existing data stores.

## Data Parallelism

A machine learning model training strategy used to maximize the utilization of compute resources (CPUs/GPUs) in which the data is distributed across two or more devices

## Data Processing

The process of altering data by means of aggregation, joins, transformations, and others.

## Data Replication

A strategy used to mitigate the potential of data loss in the event of a system or component failure. In the most basic form, it involves writing identical data to more than one device or location. More efficient techniques like erasure coding incorporate mathematics to recover lost data without referring to an explicit copy of the data.

## Database

A tool used to collect and organize data. Typically, database management systems allow users to interact with the database.

## Decision Point

Also decision rule or threshold, is a cut-off point in which anything below the cutoff is determined to be a certain class and anything above the cut-off is the other class.

## Decision Tree

A tree-based model which traverses examples down to leaf nodes by the properties of the examples features.

## Deep Learning

Optimizing neural networks, often with many hidden layers, to perform unsupervised or supervised learning.

## Dependent Variable

A variable whose variation depends on other variables.

## Derivative

Indicates how much the output of a function will change with respect to a change in its input.

## Dimensionality Reduction

The process of reducing the dimensionality of features. This is typically useful to speed up the training of models and in some cases, allow for a wider number of machine learning algorithms to be used on the examples. This can be done with SVD (or PCA) and as well, certain types of neural networks such as autoencoders.

## Dimensions
Here, the number of features associated with a particular example.

## Discriminative Model
A model which aims to approximate the conditional probability of the features and labels.

## Distributed Cache
A cache which is distributed across two or more machines

## Dot product

Here, the inner product, such that two provided vectors are element-wise multiplied followed by summing each result to produce a scalar.

## Downsampling
Removing a number of majority class examples. Typically done in addition to upweighting.

## Dropout
A regularization technique used per layer to reduce overfitting. Dropout involves randomly omitting neurons from the neural network structure at each training iteration. Effectively, dropout produces an ensemble of neural networks. Dropout is incomplete without adjusting for the dropout in preparation for prediction

## Early Stopping
Halting the gradient descent process prior to approaching a minima or maxima.

## Echo Chamber
A state of a recommendation system in which user behavior is reinforced by the recommendations themselves.

## Eigendecomposition
Applicable only to square matrices, the method of factoring a matrix into its eigenvalues and eigenvectors. An eigenvector is a vector which applies a linear transformation to some matrix being factored. The eigenvalues scale the eigenvector values.

## Elastic MapReduce (EMR)
An Amazon Web Services product which provides users access to a Hadoop cluster.

## Elbow Method
A method of finding the best value for k in k-means. It involves finding the elbow of the plot of a range of ks and their respective inertias.

## Embedding Layer
Typically used as the first layer in a neural network which is optimized to transform the input in an effort to maximize.

## Embedding Space
The n-dimensional space where embeddings exist. Typically, the embedding space can be used to generate the top-k candidates by using the k-nearest neighbors algorithm.

## Ensemble
Using more than one model to produce a single prediction.

## Epoch
One complete cycle of training on all of the examples.

## Euclidean Distance

The length of the line between two points.

## Evidence

The denominator of the Naive Bayes classifier.

## Exactly-once Semantics
Guarantees that an object within a distributed system is processed exactly once. Other semantics include maybe, at-least-once, and at-most-once

## Examples
Pairs of features and labels.

## Experiment Collision
The event where one experiment unintentionally influences the result of one or more separate experiments.

## Exploding Gradient
The repeated multiplication of large gradients resulting in an overflow, or infinity-value products.

## F1 Score
The harmonic mean of the precision and recall.

## Feature Hashing
Representing feature inputs, such as articles or messages, as the result of hashes modded by predetermined value.

## Feature Interaction
Features that are multiplied by one another in order to express relationships that can't be represented by adding the independent variable terms together.

## Feature Map
The result of applying a kernel to an image or to another feature map.

## Feature Normalization
Typically referring to feature scaling that places the values of a feature between 0 and 1.

## Feature Standardization
Typically referring to feature scaling that centers the values of a feature around the mean of the feature and scales by the standard deviation of the feature.

## Feature Transformation
A mathematical function applied to features.

## Features
A set of quantities or properties describing an observation. They can be binary like "day" and "night"; categorical like "morning", "afternoon", and "evening"; continuous like 3.141; or ordinal like "threatened", "endangered", "extinct", where the categories can be ordered.

## Featurization
The process of transforming raw inputs into something a model can perform training and predictions on.

## Flatten Layer
Takes in a series of stacked feature maps and flattens out all of the values into a single feature vector.

## Forward Pass
Calculating an output of a neural network for a particular input.

## Fully Connected Neural Network
A neural network that has every neuron connected to every other neuron in the subsequent layer; also called a dense layer.

## Gated Recurrent Unit
A type of RNN with a hidden state, update gate, and reset gate.

## Gaussian Distribution

A very common type of probability distribution which fits many real world observations; also called a normal distribution.

## Generalize
The ability of a model to perform well on the test set as well as examples beyond the test set.

## Generative Adversarial Networks
Also GANs, take advantage of a concept called adversarial min max. The generator generates fake data and tries to minimize a particular loss function and the discriminator tries to correctly identify fake data from real data in an effort to maximize a particular loss function.

## Generative Model
A model which aims to approximate the joint probability of the features and labels.

## Gini Impurity
Used as a way to determine the best split point for a given node in a classification tree. It's based on the probability of incorrectly classifying an item based on all of the items in the node.

## Gower Distance

Used to calculate the distance between two points which have mixed feature types.

## Gradient

A vector of partial derivatives. In terms of neural networks, we often use the analytical gradient in software and use the numerical gradient as a gradient checking mechanism to ensure the analytical gradient is accurate.

## Gradient Clipping
Capping the value that the gradient is allowed to be. This is typically used in an effort to avoid exploding gradients. However, initialization techniques are favored.

## Gradient Descent
An iterative algorithm with the goal of optimizing some parameters of a given function with respect to some loss function. If done in batches, all of the examples are considered for an iteration of gradient descent. In mini-batch gradient descent, a subset of examples are considered for a single iteration. Stochastic gradient descent considers a single example per gradient descent iteration.

## Graphics Processing Unit
A specialized device that has many cores, allowing it to perform many operations at a time.

GPUs are often used within deep learning to accelerate training of neural networks by taking advantage of their ability to perform many parallel computations.

Hadoop Distributed File System (HDFS)
An open-source Apache software product which provides a distributed storage framework.

## Hamming Distance

The sum of the non-matching categorical feature values.

## Hard Disk Drive (HDD)
A storage device which operates by setting bits on a spinning magnetic disk. The capacity and the read/write performance of the HDD are the main characteristics to consider when using an HDD within a particular system.

## Heuristic
An approach to finding a solution which is typically faster but less accurate than some optimal solution.

## Hidden layer
A layer that's not the input or output layer in a neural network.

## Hidden State
The output of the Recurrent Neural Network at the previous timestep.

## Hinge Loss
A loss function which is used by a soft-margin SVM.

## Hyperbolic Tangent
A symmetric activation function which ranges from -1 to 1.

## Hyperparameter
Any parameter associated with a model which is not learned.

## Hyperparameter Optimization
The process of searching for the best possible values for the hyperparameters of some machine learning model.

## Hyperplane

A decision boundary in any dimension.

## Image Kernel
A single array consisting of some defined numbers, which is applied to an image for some desired effect. Usually used in the context of image filtering.

## Image Padding
Adding a border of zero-valued pixels to an input image. The goal of padding is typically to ensure that the dimensions of an image remain the same after applying an image kernel. Libraries typically allow two options: Valid Padding, which means not to pad the image, and Same Padding, which means to zero-pad the image.

## Implicit Rating
A rating obtained from user behavior as opposed to surveying the user.

## Implicit Relevance Feedback
Feedback obtained from user behavior as opposed to surveying the user.

## In-memory Database
A database which relies either solely or primarily on the RAM of a computer.

## Independent Variable
A variable whose variation is independent from other variables.

## Inertia
The sum of distances between each point and the centroid.

## Initialization Techniques
Ways to cleverly initialize the weights of neural networks in an attempt avoid vanishing and exploding gradients. Kaiming initialization, used with asymmetric activation functions and Xavier/Glorot initialization, used with symmetric activation functions are both examples. These techniques usually depend on the fan in and fan out per layer.

## Inverse User Frequency
The added modelling assumption that if a user interacts with a less overall popular item, then the interaction should count for more.

## Irreducible Error
The error produced from the noise within the training examples which can't be eliminated with models.

## Jaccard Distance

One minus the ratio of the number of like binary feature values and the number of like and unlike binary feature values, excluding instances of matching zeros.

## Jupyter Notebook
A Project Jupyter product which provides an interactive workspace to execute code.

## K-means++
Using a weighted probability distribution as a way to find the initial centroid locations for the k-means algorithm.

## Keras

Software that acts as an interface for Tensorflow and aims to simplify the experience of working with neural networks.

## Kernel Density Estimation

Also KDE, a way to estimate the probability distribution of some data.

## Kernel Trick
The process of finding the dot product of a high dimensional representation of feature without computing the high dimensional representation itself. A common kernel is the Radial Basis Function kernel.

## L2 Loss
The sum of the squared errors of all the training examples. Not to be confused with L2 regularization.

## Labels
Usually paired with a set of features for use in supervised learning. Can be discrete or continuous.

## Laplace Smoothing

A type of additive smoothing which mitigates the chance of encountering zero probabilities within the Naive Bayes classifier.

## Learning Rate
A multiple, typically less than 1, used during the parameter update step during model training to smooth the learning process.

## Lemmatization
A more calculated form of stemming which ensures the proper lemma results from removing the word modifiers.

## LightGBM

An open-source library created by Microsoft which provides a distributed gradient boosted framework. Short for Light Gradient Boosted Models.

## Likelihood

The probability of some features given a particular class.

## Line of Best Fit
The line through data points which best describe the relationship of a dependent variable with one or more independent variables. Ordinary least squares can be used to find the line of best fit.

## Linear Activation
A symmetric activation function which assigns the output as the value of the input.

## Local Optima

A maxima or minima which is not the global optima.

## Long Short-term Memory
A type of RNN with a cell and hidden state, input gate, forget gate, and an output gate.

## Manhattan Distance

Sum of the absolute differences of two input features.

## Margin
The space between the hyperplane and the support vectors. In the case of soft margin Support Vector Machines, this margin includes slack.

## Matrix
An array of values, usually consisting of multiple rows and columns.

## Matrix Factorization
Factors the user-item matrix into embeddings such that multiplying together the resulting embedding layers gives an estimate for the original user-item matrix.

## Matrix Transpos
An operator that flips a matrix over its diagonal.

## Mcfadden's Pseudo R-squared

An analog to linear regression's R-squared which typically takes on smaller values than the traditional R-squared.

## Mean Absolute Error
The average of the absolute differences across the training examples.

## Mean Average Precision
A binary ranking metric which takes into account the relevance of ranked items with regards to their position.

## Mean Reciprocal Rank
A binary ranking metric which takes into account the first spot in a ranking which contains a relevant item.

## Mean Squared Error
The average squared difference between the prediction and true label across all examples.

## Mechanical Turk
A crowdsourcing service which performs on-demand tasks that computers are currently unable to do.

## Missing Data
When some features within an example are missing.

## MLlib
A library provided by Apache Spark which provides Spark clusters access to machine learning algorithms and related utilities. MLlib provides a Dataframe-based API which is unofficially referred to as Spark ML.

## Mode Collapse
A challenge of training GANs in which the generator generates data which successfully confuses the discriminator and so the generator exploits this vulnerability and only produces that particular data over and over. A way to mitigate this is to force the generator to see responses from the discriminator multiple timesteps ahead. One such method is called an Unrolled GAN.

## Model
An approximation of a relationship between an input and an output.

## Model Parallelism
A machine learning model training strategy used to maximize the utilization of compute resources (CPUs/GPUs) in which the model is distributed across two or more devices.

## Model Training
Determining the model parameter values.

## Momentum
A concept applied to gradient descent in which the gradients applied to the weight updates depends on previous gradients.

## Multi-Armed Bandit (MAB)
A process which provides a number of choices

## Multinomial Distribution

A distribution which models the probability of counts of particular outcomes.

## Multinomial Logistic Regression
Logistic Regression in which there are more than two classes to be predicted across.

## N-gram
A series of adjacent words of length n.

## Neuron
Sometimes called a perceptron, a neuron is a graphical representation of the smallest part of a neural network. For the Machine Learning Crash Course we may reference neurons as nodes or units.

## Non-convex Function
A function which has two or more instances of zero-slope.

## Non-differentiable
A function which has kinks in which a derivative is not defined.

## Nonlinear Regression
A type of regression which models nonlinear relationships in the independent variables.

## Norm
Here, the L2 Norm, is the square root of the sum of squares of each element in a vector.

## Normalized Discounted Cumulative Gain
An information retrieval metric which assigns a value of a particular ranking based on each item's position and relevance.

## Numba
A just-in-time Python compiler which resolves a subset of the python programming language down to machine code

## Odds Ratio

The degree of associate between two events. If the odds ratio is 1, then the two events are independent. If the odds ratio is greater than 1, the events are positively correlated, otherwise the events are negatively correlated.

## OLAP
Online analytical processing. A system that handles the analytical processes of a business, including reporting, auditing, and business intelligence. For example, this may be a Hadoop cluster which maintains user subscription history for Netflix. This is opposed to OLTP.

## OLTP
Online transaction processing. A system that handles (near) real-time business processes. For example, a database that maintains a table of the users subscribed to Netflix and which is then used to enable successful log-ins would be considered OLTP. This is opposed to OLAP.

## One-hot Encoding
An encoding for categorical variables where every value that a variable can take on is represented as a binary vector

## Online Learning
Incremental learning within a model to represent an incrementally changing population.

## Optimizers
Techniques which attempt to optimize gradient descent

## Orthogonal

Perpendicular is n-dimensions.

## Outlier
A feature or group of features which vary significantly from the other features.

## P-value

The probability of finding a particular result, or a greater result, given a null hypothesis being true.

## Parallelization
When two or more computer programs are executed at the same instant across more than one processor.

## Parameter
Any trained value in a model.

## Parameters
Also, weights, or coefficients. Values to be learned during the model training.

## Parquet
A column-oriented data storage format provided by Apache.

## Partial Derivative

The derivative of a function with respect to a single variable.

## Pearson Correlation

A measure of the correlation between two inputs. In the context of recommendation systems. Pearson correlation can be used to construct an item-item similarity matrix.

## Polynomial
A function with more than one variable/coefficient pair.

## Pooling
Most often max pooling and sometimes average pooling, in which a kernel is applied to an image and the max or average of the pixel values in the kernel overlay is the resulting pixel value in the output image.

## Posterior
The probability of a class given some features.

## Pre-trained Models
Models which have already been trained. These trained models can be used as layers (like embedding layers) of not-yet-trained neural networks to increase performance of these neural network.

## Precision
The number of true positives divided by the true positives plus false positives.

## Presentation and Trust Bias
Biases found within ranking which arise from the placement of items within a ranking.

## Principal Component Analysis
Also PCA, is eigendecomposition performed on the covariance matrix of some particular data. The eigenvectors then describe the principle components and the eigenvalues indicate the variance described by each principal component. Typically, SVD is used to perform PCA. PCA assumes linear correlations in the data. If that assumption is not true, then you can use kernel PCA.

## Prior

Indicates the probability of a particular class regardless of the features of some example.

## Probability

How likely something is to occur. This can be independent, such as the roll of the dice, or conditional, such as drawing two cards subsequently out of a deck without replacement.

## Probability Distribution
A function that takes in an outcome and outputs the probability of that particular outcome occurring.

## Pruning Neurons
Removing neurons from a neural network in an effort to reduce the number of model parameters if by removing the neurons, equivalent performance can be obtained.

## R-squared

Also the coefficient of determination, the percent of variance in the dependent variable explained by the independent variable(s).

## Random Access Memory (RAM)
A device on a computer which stores the data and machine code of a running computer program.

## Random Forest
An ensemble technique which trains many independent weak learners. This generally reduces the variance error.

## Rank r Approximation
Using up to, and including, the rth terms in the singular value decomposition to approximate an original matrix.

## Ranking
Optimizing machine learning models to rank candidates, such as music, articles, or products. Typically, the goal is to order the candidates such that the candidates which are most likely to be interacted with (purchased, viewed, liked, etc.) are above other candidates that aren't as likely to be interacted with.

## Recall
Also sensitivity, is the proportion of true positives which are correctly classified.

## Receiver Operator Characteristic Curve
Also ROC curve, is a plot of how the specificity and sensitivity change as the decision threshold changes. The area under the ROC curve, or AUC, is the probability that a randomly chosen positive example will have a higher prediction probability of being positive than a randomly chosen negative example.

## Receptive Field
The number of neurons in a preceding layer which are connected to an adjacent layer of neurons. The largest receptive field is a fully-connected neural network.

## Recommendation Carousel
A component within a graphical or conversational user interface which presents recommendations to a user. This can include products, ads, and media content.

## Recommendation Systems
Systems with the goal of presenting an item to a user such that the user will most likely purchase, view, or like the recommended item. Items can take many forms, such as music, movies, or products. Also called recommender systems.

## Rectified Linear Unit
An asymmetric activation function which outputs the value of the positive inputs and zero otherwise. There are variations such as the Leaky ReLU. They can be susceptible to the dead neuron problem but generally perform well in practice.

## Recurrent Neural Network
Also RNN, a neural network in which the output is routed back into the network to be used with the subsequent input.

## Regularization
A technique of limiting the ability for a model to overfit by encouraging the values parameters to take on smaller values.

## Residuals
The distance between points and a particular line.

## Sample Selection Bias
The bias that occurs when sampling a population into one or more subgroups at random results in a systematic inclusion or exclusion of some data.

## Sample Size
The number of observations taken from a complete population.

## Scalar

A single value, as opposed to a vector.

## scikit-learn
A machine learning python library which provides implementations of regression, classification, and clustering.

## Seasonality
The predictable changes of data throughout the calendar year.

## Sensitivity
Also recall, is the proportion of true positives which are correctly classified.

## Session ID
A unique identifier assigned to a user to keep track of a user's connected interactions. For instance, a session may include a user logging in, purchasing an item, and logging out. Here, the session ID would be used to reference the group of those three interactions. This session ID can be stored in the user's internet browser as a cookie.

## Shadow Test
Running two or more versions of software in parallel while only surfacing the result of one of the experiences to the end user. This is done in an effort to gauge the differences between the software versions.

## Shift Invariance
One of the goals of a convolutional neural network: objects within an image can be in different areas of the image yet still be recognized.

## Shilling Attack
A type of attack on a recommendation system in which users manipulate the recommendations by inflating or deflating positive interactions for their own or competing items.

## Shrinkage
A learning rate for gradient boosted trees.

## Side Features
Features in addition to item and user embeddings. This can include properties of items and users.

## Sigmoid Function
Also the logistic function, a function which outputs a range from 0 to 1.

## Silhouette Method
A method of finding the best value for k in k-means. It takes into account the ratios of the inter and intra clusters distances.

## Simple Matching Distance

One minus the ratio of the number of like binary feature values and the number of like and unlike binary feature values.

## Simpson's Paradox
When a pattern emerges in segments of examples but is no longer present when the segments are grouped together.

## Singular Value Decomposition
Also SVD, a process which decomposes a matrix into rotation and scaling terms. It is a generalization of eigendecomposition.

## Slack
The relaxing of the constraint that all examples must lie outside of the margin. This creates a soft-margin SVM.

## Softmax
A sigmoid which is generalized to more than two classes to be predicted against.

## SparkML
Refers to APIs which provide machine learning capabilities on Spark dataframes.

## Specificity
The proportion of true negatives which are correctly classified.

## Split Point
A pair of feature and feature value which is assigned to a node in a decision tree. This split point will determine which examples will go left and which examples go right based on the feature and feature value.

## Standard Deviation

The amount of deviation around the mean for a particular distribution

## statsmodels

Python module which provides various statistical tools.

## Stemming
Removing the ending modifiers of words, leaving the stem of the word.

## Stop Word
A word, typically discarded, which doesn't add much predictive value.

## Stride
The incremental rate at which a kernel is applied to an image or feature map.

## Sub-gradient
The gradient of a non-differentiable function.

## Supervised Learning
Optimizing machine learning models based on previously observed features and labels. Typically, the goal is to attach the most likely label to some provided features.

## Support Vectors
The most difficult to separate points in regard to a decision boundary. They influence the location and orientation of the hyperplane.

## Surrogate Split
A suboptimal split point reserved for examples which are missing the optimal split point feature.

## TF-IDF
Short for Term Frequency-Inverse Document Frequency, TF-IDF is a method of transforming features, usually representing counts of words, into values representing the overall importance of different words across some documents.

## Time Decay
The added modelling assumption that interactions between items and users should count for less over time.

## Tokenization
The splitting of some raw textual input into individual words or elements.

## Unbalanced Classes
When one class is far more frequently observed than another class.

## Uniform Distribution

A probability distribution in which each outcome is equally likely; for example, rolling a normal six-sided die.

## Unsupervised Learning
An approach within machine learning that takes in unlabeled examples and produces patterns from the provided data. Typically, the goal is to discover something previously unknown about the unlabeled examples.

## Upweighting
Increasing the impact a minority class example has on the loss function. Typically done in addition to downsampling.

## User Agent
An identifier used to describe the software application which a user is using to interact with another software application. For instance, an HTTP request to a website typically includes the user agent so that the website knows how to render the webpage.

## User-item Matrix
A matrix which contains the interactions of users and items. Items can take the form of products, music, and videos.

## Validation
The technique of holding out some portion of examples to be tested separately from the set of examples used to train the model.

## Vanishing Gradient
The repeated multiplication of small gradients resulting in an underflow, or 0-value products.

## Variance

The standard deviation squared.

## Variance Inflation Factor

A measure of multicollinearity in a regression model.

## Vector

Here, a feature vector, which is a list of features representing a particular example.

## Vectorizer
Used in a step of featurizing. It transforms some input into something else. One example is a binary vectorizer which transforms tokenized messages into a binary vector indicating which items in the vocabulary appear in the message.

## Vocabulary
The list of words that the Naive Bayes classifier recognizes.

## Weak Learner
Shallow decision trees in our case. However, it generally can be any underfitting model.

## XGBoost

An open-source library which provides a gradient boosted framework. Short for Extreme Gradient Boosting.

## Zookeeper

A service designed to reliably coordinate distributed systems via naming service, configuration management, data synchronization, leader election, message queuing, or notification systems.
Variance Inflation Factor
