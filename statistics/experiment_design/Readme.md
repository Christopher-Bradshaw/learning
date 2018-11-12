# Experiment Design

We want to model some surface. This might be the likelihood as a function of some parameters, or clustering as a function of cosmology, or etc... To do this we take some observations at chosen points in parameter space. But how do we chose those points to maximize the amount we learn about the surface?

Before starting, we note the "effect sparcity" principle. This claims that a few dimensions will dominate the information content, or that many dimensions will be uninformative. As we can't know what these are before running the experiment we should have a design that projects well into subspaces.

Resources:
* [A nice summary of many of these designs](https://www2.isye.gatech.edu/~jeffwu/isye8813/spacefilling_designs.pdf).
* [PYDOE: A Python library implementing some of these designs](https://pythonhosted.org/pyDOE/)


## Full Factorial

Assume each of our dimensions has discrete possible values (e.g. the only options are [on, off], [0, 1, 2], etc). Then we can cover all possible options with `num_possibilities ^ num_dimensions` trials.

For continuous variables this is analagous to a grid design where we use for example 10 points per dimension and our number of points scales as `10 ^ num_dimensions`.

This scales exponentially in dimension and so if not feasible in high dimension problems.

## Maximin

Try to maximize the minimum distance between points. By ensuring that points are well spread out we hope to cover the space well.

However, in doing this we don't consider projections at all and so if one dimension is uninformative we probably have a large number of redundant points.

## Minimax

Try to minimize the maximum distance to a point. By ensuring that everywhere in our parameter space is near a sample we can predict any region well.

As above, no consideration of projections. Likely will have redundant points


## Latin Hypercube

See [the paper](http://cybermesa.com/~michaelm/Papers/Sampling/Techno-79.pdf) and the useful [wikipedia page](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)

Ensures no redundancy, but can be poor space filling (e.g. all points on the diagonal). Can combine with Maximin to create better space filling will still good projections!

However the uniform projection is only in a single dimension. Projections down to a smaller subspace might be less good.


## Maximum Projection (MaxPro)

See [the paper](https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/max_projection_designs_roshan2015.pdf) but this has both good space filling and projection down to any subspace.
