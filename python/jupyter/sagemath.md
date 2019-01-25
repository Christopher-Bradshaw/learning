To get a sagemath kernel, you first need to have installed sage. Obviously...

Then, create a `sagemath` directory in `~/.local/share/jupyter/kernels` which should already exist, and contain other kernels. Inside that, create a `kernel.json` file that contains,

```
{
 "argv": [
  "sage",
  "-python",
  "-m",
  "sage.repl.ipython_kernel",
  "-f",
  "{connection_file}"
 ],
 "display_name": "SageMath",
 "language": "sage"
}
```

And now it should appear in the drop down list of kernels!
