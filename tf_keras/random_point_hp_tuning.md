# Hyperparameter Tuning with Random Points

Tuning hyperparameters with random points is usually better than with a grid.

### A  Grid Search

While tuning hyperparameters, it is common to see a grid function used to tune a model. The end user specifies a list of possible parameter values and these are tested in a logic like this:

```
## specify parameter values.
learning_rate_chioces = [.0001, .001, .01, .1, 1]
alpha_choices = [.0001, .001, .01, .1, 1]

for lr in learning_rate_choices:
    for alpha in alpha_choices:
        ## pass the combinantions of parameters and score.
        score_model_with_params(lr, alpha)
```

### A Random Point Search

But, following a comment by Andrew Ng from his DeeplearningAI course, searching random points in the same space is a superior method. The logic of random point searching looks like this:
```
## Define a function that finds numbers in range 10^(-3x-1)

alpha_choice = (lambda x: 10**(-3*x)/10) 
n = 25 ## Let's sample 25 models from the space.
for i in range(0,n):

    ## generate a random number x on the interval (1e-4,1e-1)
    ## space so log10(x) is linear.
    random_numbers = np.random.rand(2)
    alpha = alpha_choice(random_numbers[0]) 
    lr = _choice(random_numbers[1])

    ## pass those to score the model with them.
    score_model_with_params(lr, alpha)

```
### The Random Point search is better

 Why? There are two reasons. If we run the grid search on 10 values for each parameter, we run 100 searches of the model on 10 values of lr and 10 values of alpha. Suppose we run the point search 100 times: we run 100 searches on 100 values of alpha and 100 values of lr. That's a lot more information about the parameters, but still distributed in the same space as the grid. Thus, if alpha makes a big difference but lr doesn't, we learn a lot about best choices for alpha. 

 Second, what if 100 searches is a lot of computation? Instead, we could run 25 searches of the random point sort, identify the region in which the parameters did the best and then search 25 more times in newly identified region. The complexity of a grid search grows geometrically with the number of parameters searched. The point search can be grown linearly by adjusting the number of searches, and then repeated once to find the right parameters in the parameter subpace that looks best. Thus, we can reduce computational demands with the point search.

 (By the way, the numbers in these examples are intended to be for illustration. Typically, one works with sizes )

 ## Can we implement this with Keras tuner?

 Yes we can! The process looks a little different from above. Above, we randomly select numbers from a space and build the models with those random numbers. Keras builds the space of models first and then selects a tuning process to apply to those models. The RandomSearch tuner will implement the above process by randomly selecting the models. To be a little more precise: Keras tuner uses a Hypterparamters object that specifies possible values for a hyperparameter and then selects them at random during a random search.

### Example with Keras Tuner

 Imagine a relatively simple example in which we're tuning a single layer with L2 regularization and we want to tune the number of units in the layer and the L2 alpha. The inputs have 784 features. We can construct a model builder like this:

 ```
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt

 ## specify a large range with a suitable distribution of features.
 hp_units = hp.Int(min_value = 40, 
                max_value = 100, 
                step = 1, 
                sampling = 'linear')

 hp_alpha = hp.Float(min_value = 10e-4,
                    max_value = 10e-1,
                    step = .00001,
                    sampling = 'log')
```
These hyperparamter objects specify a range of integers and a range of floating points. The ```sampling``` argument tells keras what probability distribution to use for the selection. 'linear' means that they're equally probably, log means that each order of maginitude should be equally likely to be selected. (i.e., there will be as many in the neighborhood of 10e-4 as in the neighborhood of 10e-1.)
```
 ## set up a model with those as values for hypermodel parameters.
 def model_builder():
    inputs = keras.Input(shape=(784))
    x = layers.Dense(hp_units,
                    kernel_regularizer=keras.regularizers.l2(hp_alpha)
                )(inputs)
                    
    output = layers.Dense(1, activation='sigmoid')
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model
```
Once we've established the space of the hyperparamters and the model builder that takes those as arguments, we need to create a RandomSearch tuner:
```
tuner  = RandomSearch(model_builder, 
                    max_trials = 25
                    )
tuner.search(epochs = 40)
```






