# Cluster setup

I think it is important to play with MPI in a cluster setting. I think that I might be missing some things if I run everything locally.

Ideally I would be able to run MPI code shared between my laptop and desktop but that failed.

Second best would be to run between my desktop and a remote machine but that failed too.

Most annoying would be to need to run it on multiple servers, but that is what it seems I need to do.

We start with the quickstart guide on how to get MPI running on multiple GCP instances + then some things I learned while failing to run things locally.

## Using GCP Instances

Useful reading: [mpi on gcompute](https://www.reidatcheson.com/c/programming/hpc/mpi/cloud/2015/12/27/mpi-on-gcompute.html).

My quickstart guide:

* Create a new project! Add your ssh key to the project ssh keys.
* Spin up 1 gcompute instance (cheapest are < $5 per month). Notes that follow assume you use Debian.
* ssh onto that instances using the public IP. **N.B.** use the GCP console ssh to work out what the username that has been created is!
    * Install openmpi `sudo apt-get install libopenmpi-dev`
    * Create a directory `~/bin`
    * Create new ssh keys `ssh-keygen`. Once this is done, add this ssh key to the project ssh keys.
        * This key will allow the GCP machines to ssh to each other.
* Through the cloud console take a snapshot of this image.
* Spin up a second (or more!) machine using the snapshot. Any machine you launch with the snapshot will have openmpi/the ssh key/`~/bin` already created.
* Build your mpi code locally `mpicc ...`.
* Copy the built code onto each of the machines `scp binary username@host_ip:~/bin/`
* Create a hosts file with the internal IPs (or instance names). Copy this to one of the machines.
* ssh onto that machine. Run the code with `mpirun -n X --hostfile hosts ~/bin/binary`


## Using my laptop + desktop (Failed)

Theoretically you can just create a hosts file and run `mpirun -n 4 -hostfile hosts a.out` and things will just work but predictably they didn't. Some issues:

`bash: orted: command not found`: MPI starts a non-interactive shell on the remote machine and so changes to my path in `~/.bash_profile` to include the MPI binaries weren't being included. Moving this to `~/.bashrc` fixed it.

We then get a `A process or daemon was unable to complete a TCP connection to another process`. My guess here is that the remote process couldn't get back to my laptop. Let's spin up a server somewhere and retry.

## Using my desktop + Digital Ocean droplet (Also failed)

Switching over to launching from the desktop with the remote machine now in the cloud.

On the remote note we then get `mpirun was unable to launch the specified application as it could not access or execute an executable:`. I was assuming that you launched the code from one machine it would copy the binary over. It doesn't. I guess this makes sense - what if the nodes have different architectures and you want to compile it differentely? Or even (crazy idea), what if you want different code running on different nodes! Make sure that the binary is on all machines, in the same place (or just in your path).

```
$ mpirun -n 4 --hostfile hosts a.out
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Hello world from processor cb.ucsc.edu, rank 1 out of 4 processors
Hello world from processor cb.ucsc.edu, rank 0 out of 4 processors
Hello world from processor fedora-s-1vcpu-1gb-sfo2-01, rank 3 out of 4 processors
Hello world from processor fedora-s-1vcpu-1gb-sfo2-01, rank 2 out of 4 processors
```

That was remarkably painless. However, let's now try run send_and_receive. Here we run into problems. We are unable to send to the remote machine. A list of things I tried:

1. Ensure that you can ssh from both to both
2. Ensure that ports are open

My final `mpirun ...` looked something like:
```
	mpirun -v -n 4 \
     --hostfile hosts \
     -mca btl self,tcp \
     -mca btl_tcp_if_exclude virbr0,lo \
     -mca btl_tcp_port_min_v4 15000 \
     -mca btl_tcp_port_range_v4 100 \
     send_and_receive
```
And I did a lot of playing with IPTABLES and firewall-cmd. None of which was fun and none of which worked. I've given up on this and will just read on and hopefully come across an explanation. Or maybe this is impossible? I don't know... Fortunately doing it fully remotely works fine.
