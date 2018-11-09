# Stan

## What is Stan?

Stan is a imperative probabilistic programming language. [Imperative](https://en.wikipedia.org/wiki/Imperative_programming) I am good with, but what is probabilistic? From the [reference manual](stan-reference-2.17.0.pdf):

> Stan is a probabilistic programming language in the sense that a random variable is a bona fide first-class object

For the moment I'm going to assume that first class support for random variables/distributions makes something probabilistic (maybe in the sense that functional programming requires first class functions?).

Also note that **Stan is a programming language**. It is not a library/module/etc.

## What can you do with Stan?

Optimization + Sampling are what I have done so far.

**N.B.** these are things that you can do without stan (e.g. with scipy.optimize and your favourite MCMC library). However, I expect that value prop of stan is to make doing these things easy, fast and reliable.

## Installing Stan

If you are hitting it through python, install with `pip install pystan`.

If you want to run it through the command line, install [cmdstan](https://github.com/stan-dev/cmdstan/wiki/Getting-Started-with-CmdStan)

## Running Stan

1) Write some stan code.
2) Compile `my_model.stan -> C++ -> binary` using cmdstan/or the python interface.
3) Run the binary (or the returned python object), passing any data it needs.

## Language details

### Types

Stan has two primitive data types: `real` (internally 64 bit) and `int` (internally 32 bit).
Container data types also exist: e.g. `vector`, `matrix`. These **are indexed from 1**. Also note that the container sizes are not validated at compile time.
Variables can be constrained (e.g. to be positive).

Example set of definitions
```
real theta;	                            // declares unconstrained real
int<lower = 0> x;                       // declares int with lower bound
vector<lower = -1, upper = 1>[3] rho;   // declares vector of length 3 with lower and upper bounds
real sum = 0                            // declares and assigns real
```

### Blocks

A Stan program is separated into sections or blocks. All these blocks are optional, but if they exist must appear in this order:

```
functions {
    // Define user defined functions
}
data {
    // Declare the data variables that is needed in the model
    // When running the Stan program these variables must be provided as an argument
}
transformed data {
    // Define constants and transformations of the incoming data
}
parameters {
    // Declare the free parameters in the model. This is what will be sampled/optimized
    // Only declarations are allowed here - you cannot assign to free parameters.
}
transformed parameters {
    // Define transformed versions of the parameters (potentially using the data).
    // I think the reason you might want to do this is to make the model simpler.
}
model {
    // Compute the log probability
}
generated quantities {
    // ???
}
```

Pretty much all variables have global scopes (i.e. not scoped to their block - that would make no sense). Variables defined in functions are scoped to that function. Arguments to functions are `const`.

### Sampling Statement and `target`

An important feature of stan is the sampling statement that looks like:

```
y ~ normal(mu, sigma);
```

This computes the log liklihood that the data vector y is described by a normal distribution with mean mu and sd sigma. It adds this to a "variable" called `target` which is initilialised to 0. This can be assigned to directly (e.g. `target = 0`, `target += 4`) but cannot be used as a variable (e.g. `print(target)`). To access it use the function `target()`.

Note that future samplings will add to `target` (e.g. if you just run that sampling twice `target` will be double than if you ran it once).
