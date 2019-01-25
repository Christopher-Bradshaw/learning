import autograd.numpy as np
from autograd import grad, jacobian, hessian

from line_search import get_line_length

assert "autograd" in np.__file__ # Need to make sure that we get the correct numpy - my config/other imports might import the default one.



f = lambda x: x[0]**2 + x[1]**2
grad_f = grad(f)

x0 = np.array([2, 2], dtype=np.float32)

print(get_line_length(f, grad_f, x0, -x0, 10))
