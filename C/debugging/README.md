# Debugging with GDB

Some guides, [beej's quickstart](https://beej.us/guide/bggdb/)

## Quickstart
* Compile the code with `-g`
* Start the debugger `gdb a.out`
* Insert breakpoints `b 42`
* Run the code, potentially passing args/stdin `run arg1 arg2 < some_file`

If you are debugging python, the executable is just the python interpreter. Make sure that the C libraries you are debugging have been compiled with debug flags.

e.g. `gdb python3` then `run program.py args`

## Post Crash (w GDB)

You ran code with `gdb` and it crashed. What now?

Backtrace, `bt`, to see how the frame stack at the time of the crash. This will show you all the function calls and their arguments! Such a quick way to isolate the place where it crashed.

Once you know which functions all the stack frames correspond to you can also move between then with `up`, `down`, `frame X`. Once in the correct frame you can print variables to work out the state within that frame.

## Post Crash (w/o GDB)

If you were running without GDB you probably saw a log line like `Aborted (core dumped)`. This corefile contains info about the memory of the program that can be inspected by GDB to learn about the state of the program when it crashed. But where is it?

Where it is depends on your OS. It might just be in the local dir, but it more likely has been sent to an automatic reporting/logging service. On Fedora this is `coredump`. To get access to one:

* Use `coredumpctl` to see all coredumps. Find the PID of the one you care about
* `coredumpctl dump <PID> --output <SOMEFILE>
* `gdb /path/to/binary <SOMEFILE>

And now you are in the same state as the post crash w GDB world.

## Gotchas

You were debugging and added comments to the source as you went. Now things have got our of sync and gdb isn't printing what it is actually executing. run `directory`. See [here](https://stackoverflow.com/questions/4118207/how-to-reload-source-files-in-gdb).

## MPI

The best way I have found for this is to add the following function where you want to start debugging from. This will log `PID <num> on <machine> ready for attach`. Enter the debugger with `gdb /path/to/executable <PID>`. Note that if you are running python the executable will just be python. Once in there, go up a couple of frames (out of sleep/_spin_for_gdb/etc) until you get to the func you want to debug. Add a breakpoint. Come back down the stack to this function and `set var i=1`. Then continue and you should hit your breakpoint.
```
void _spin_for_gdb() {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
        sleep(5);
```
