# NERSC

Getting started on NERSC (possibly applicable to other supercomputing platforms).


## Shifter

Basically docker with some features that make it work better in a cluster environment?
If you drop into a bash shell in a shifter image (`shifter --image $IMG_NAME bash`) you are in the image filesystem, but also have access to some cluster things (e.g. `/global`)


## Scratch

At least on Cori, this is located at `/global/cscratch1` (I'm guessing the `c` stands for cori?) and is a high performance data storage location. It is accessible from all nodes and designed to be used while running jobs. Data gets purged, so you need to copy data you want to keep onto long term storage.

See your scratch directory with the env var `$SCRATCH`

See, [Cori-scratch](https://docs.nersc.gov/filesystems/cori-scratch/)
