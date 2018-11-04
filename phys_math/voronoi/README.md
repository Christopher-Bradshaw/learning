# Voronoi Tesselation

When people mention these they always sounds incredibly cool and complicated. I think it is the Russian name... Incidentally, Georgy Voronoi studied under Andrey Markov (of Markov chains)!

They are actually pretty simple to understand!

## Description

Given a region of space and a set of seed points. The Voronoi tesselation assigns to each point the region of space for which that point is nearer than any other point. Therefore we have dividing lines equidistant betweeen two points and vertices equidistant from three points.

A really pretty visualization imagines the regions as growing circles ([source](https://en.wikipedia.org/wiki/File:Voronoi_growth_euclidean.gif)).

![Voronoi Growth](./voronoi_growth.gif)


## Alternate Metrics

In the image above we use the Euclidian distance between points. However, alternate distance metrics can be used.
