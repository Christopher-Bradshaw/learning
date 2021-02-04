# Performance Notes

* Raising to that power is slow. Use `x * x` rather than `x**2`.
* `Math.round(x)` is slow. Use `~~(x + 0.5)`.
