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

 Why? There are two reasons. Suppose we run the grid search on 5 values for each parameter, we run 25 searches of the model on 5 values of lr and 5 values of alpha. Suppose we run the point search 25 times: we run 25 search on 25 values of alpha and 25 values of lr. That's a lot more information about the parameters, but still distributed in the same space as the grid. Thus, if alpha makes a big difference but lr doesn't, we learn a lot about best choices for alpha. 

 Second, what if 25 searches is a lot of computation? Instead, we could run 10 searches of the random point sort, identity the region in which the parameters did the best and then search ten more times in that region. The complexity of a grid search grows geometrically with the number of parameters searched. The point search can be grown linearly by adjusting the number of searches, and then repeated once to find the right parameters in the parameter subpace that looks best. Thus, we can reduce computational demands with the point search.

 ## Can we implement this with Keras tuner

 Yes we can! Do to so, we need to subclass ```kerastuner.engine.hyperparameters.HyperParameters()```. Our sytax to use it will be:
 ```
 HyperParameters.RandSpace(name,func,count,d_type='float',parent_name,parent_values)
 ```



 
 What I'm imaginig is something where we can define a dictionary of functions. These functions would produce values in the right ranges, given random numbers from (0,1). The we specify the number of trials to run, pass in a hypermodel builder and the dictionary to a tuner and let it do the rest of the work, giving us the nicely scored output of other tuners from Keras.

 Something like this:

