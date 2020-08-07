### NLP Specialization 1

# Notes made on Typora

## Week 1

Concepts covered

> Classification
>
> Vector Spaces

To solve problems like

> sentiment analysis
>
> word translation

* Supervised Machine Learning
  * features X
  * Labels Y
  * Output Yhat
  * Cost function measures difference Labels and Outputs,
  * Then parameters are updated to reduce Cost Function

>Sentiment Analysis

* Build vocabulary(Size V)
* To create a sparse representation assign 1 or 0 based on whether the word appears in the target sentence or not(Creates a vector of size V)
* Building a negative and positive frequency dictionary(which maps a word and a class to the number of times that word appears in that corresponding class)
* Using the above created dictionary, we create a 3-dimensional vector with [bias, sum of +ve frequencies, sum of -ve frequencies] for an arbitrary sentence in the corpus.
* Preprocessing- Stemming and Stop words
* Preprocessing for sentiment involves removing twitter handles, urls, punctuations (maybe),  stop words(and, is etc)
* Stemming is reduces vocabulary size(by attaching words to it's stem word)
* the last step of preprocessing is lower casing

>General Overview

* tweet --> preprocessed tweet --> 3 D vector with bias, +ve, -ve frequencies
* We create an X matrix of all our vector inputs which are then fed into the logistic regression classifier

>Logistic Regression
$$
><img src="https://render.githubusercontent.com/render/math?math=h(x^{(i)},\theta) = \frac{1}{1+ e^{-\theta^{T}x^{i}}}">
$$

* x^{i} is the i-th tweet vector
* Theta is parameter to be trained. After training:
* Depending on the output of the function, if above or below a threshold of 0.5 the tweet is predicted as +ve or -ve

> Cost Function

$$
<img src="https://render.githubusercontent.com/render/math?math=J(\theta) = -\frac{1}{m}\Sigma_{i=1}^{m}[y^{(i)}\log{h(x^{i},\theta})+(1-y^{(i)})\log{(1-h(x^{i},\theta}))]">
$$

## WEEK 2

Outline

* Probabilities
* Baye's Rule
* Naive Bayes classifier
* Laplacian Smoothing
* Log Likelihood

> Probability

$$
<img src="https://render.githubusercontent.com/render/math?math=P(A \cup B) \ and P(A \cap B)">
$$

> Bayes Rule

$$
<img src="https://render.githubusercontent.com/render/math?math=P(X \mid Y) = P(Y \mid X) * \frac {P(X)}{P(Y)}">
$$

> Naive Bayes

* Create a table of probabilities for each word with it's frequency in positive and negative labels
* Each individual ratio is the sentiment for a corresponding word in the tweet

$$
<img src="https://render.githubusercontent.com/render/math?math=\Pi^{m}_{i=1} \frac{P(w_{i} \mid pos)}{P(w_{i} \mid neg)}">
$$

* the product we calculate using the above formula is greater than 1 for +ve tweets and less than 1 for -ve tweets

> Laplacian Smoothing

* Technique to avoid probabilities being zero
* The formula above is for two classes {Positive, Negative}

$$
<img src="https://render.githubusercontent.com/render/math?math=which \: is \: {P(w_{i} \mid class)} = \frac{freq(w_i,class)}{N_{class}}">

$$

* this formula is modified

$$
Laplacian \: Smoothing: \: {P(w_{i} \mid class)} = \frac{freq(w_i,class)+1}{N_{class}+V}
$$

* where 1 is added in the numerator to avoid 0 probability
* V is the number of unique words in the class(Used in this weeks assignment) to make sure that the sum of all probabilities is 1

> Log Likelihood

$$
<img src="https://render.githubusercontent.com/render/math?math=Prior = \frac{P(Pos)}{P(Neg)}">
$$

$$
<img src="https://render.githubusercontent.com/render/math?math=Complete \:Naive \: Bayes\: Formula: \frac{P(Pos)}{P(Neg)}\Pi^{m}_{i=1} \frac{P(w_{i} \mid pos)}{P(w_{i} \mid neg)} > 1">
$$

* these products run the risk of underflow(won't be stored on your computer if they are too small)
* hence we apply Log of the above score

$$
<img src="https://render.githubusercontent.com/render/math?math=Log \:Likelikhood \:Naive \: Bayes\:: \log(\frac{P(Pos)}{P(Neg)}\Pi^{m}_{i=1} \frac{P(w_{i} \mid pos)}{P(w_{i} \mid neg)} )">
$$

> Training Naive Bayes

To train Naïve Bayes model you need to:

1. Annotate the dataset with positive and negative tweets
2. Preprocess the tweets
   1. Lowe-case
   2. Remove punctuation, urls, names
   3. Remove stop words
   4. Stemming: reducing words to their common stem
   5. Tokenize sentences: splitting the document into single words or tokens.
3. computing the vocabulary for each word and class: freq(w,class)
4. get probability for a given class by using the Laplacian smoothing formula: P(w|pos),P(w|neg)
5. Compute λ(w), log of the ratio of your conditional probabilities
6. Compute log(prior)=log(P(Pos)/P(Neg))



## Week 3

> Vector Spaces

* Applications are in the domain of tasks such as question answering, paraphrasing, and summarization.
* Creating a co-occurrence matrix(with a certain distance = k)
* We use a word by word or a word by document design to count the co-occurrence 
* Then we create a vector space to check similarity 

> Euclidean Distance

* For a n-dimensional vector

$$
<img src="https://render.githubusercontent.com/render/math?math=d(v,w) = \sqrt{{{\Sigma_{i=1}^n}(v_i}-{w_i})^2}">
$$

* Norm of the compared vectors 

> Cosine Similarity

* Cosine similarity isn't biased by the size of the representations and is usually a better similarity metric than Euclidean Distance
* Given two vectors(v,w), the dot product is computed and divided by the norm of the two individual vectors

$$
<img src="https://render.githubusercontent.com/render/math?math=cos(\alpha) = \frac{v.w}{{\left\|{v}\right\|\left\|{w}\right\|}}">
$$




* Cosine value closer to 1 indicates high similarity and Cosine value closer to 0 indicates dissimilarity 

> Manipulating word vectors

* Using a vector space model of words which has captured the relative meaning of words we can manipulate the vectors to draw useful information

> Principal Component Analysis

* Used for visualization for high dimensional vectors
* This algorithm performs dimensionality reduction
* While retaining as much information as possible
* The Algo is:
  * Mean Normalize Data
  * Get Covariance matrix
  * Perform SVD(Singular Value Decomposition)
* Better explained in the notebook.

## Week 4

> Machine Translation
>
> * language translation
>
> * Calculate word embeddings for both languages and make a list of them
>
> * Transform the word embedding from one language's vector space to the other vector space
>
> * Then you identify the closest embedding in the new vector space
>
> * The transformation is performed using matrix multiplication
>
> *
> <img src="https://render.githubusercontent.com/render/math?math=X - 1st\ language\ vector\\">
> <img src="https://render.githubusercontent.com/render/math?math=Y - 2nd\ language\ vector\\">
> <img src="https://render.githubusercontent.com/render/math?math=R - Transformation\ matrix\\">
> <img src="https://render.githubusercontent.com/render/math?math=XR = Y">   
>   
>   
>  
>
> * $$
> <img src="https://render.githubusercontent.com/render/math?math=Loss = ||XR-Y||_F\\">
> <img src="https://render.githubusercontent.com/render/math?math=Gradient\ Equation- g = \frac{d}{dR}{Loss}\\">
> <img src="https://render.githubusercontent.com/render/math?math=Update\ Equation\ - R =  R -\alpha{g}">
>   $$
>
> * $$
><img src="https://render.githubusercontent.com/render/math?math=||A||_F = \sqrt{\Sigma_{i=1}^{m}\Sigma_{j=1}^{n}|a_{ij}|^2}\\">
><img src="https://render.githubusercontent.com/render/math?math=Where\ A\ is\ m*n\ matrix">   
>   $$
>
>   
>
> Document search
>
> > Learning Objectives for the above tasks
> >
> > * Transform Vectors
> >   * As explained in equations above
> > * K nearest neighbors
> >   * Once we have transformed the word vector  we use KNN to find the closest match to the transformed word
> > * Hash Tables
> >   * Hash tables allows us to create subsets of our data in order to find the nearest neighbors 
> >   * Create n-buckets
> >   * Create a dictionary with empty values as your hash table
> >   * Using a hash function assign each value to a key(bucket) in the dictionary
> > * Divide Vector spaces into regions
> > * Locality sensitive Hashing
> >   * This method reduces computational costs in finding KNN
> >   * Planes help us divide the vector space based on location
> >   * The dot product with a normal of the plane tells us whether a vector is towards one side or the other or on the plane itself
> >   * Given multiple planes we assign a hash value of 1 for +ve dot product and 0 for -ve dot product
> >   * Then we multiply the hash values to the powers of 2 and add them to obtain a single hash value
> > * Approximated nearest neighbors 
> >  

> Transforming word vectors







