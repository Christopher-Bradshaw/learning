import autograd.numpy as np
from .line_search import get_line_length
from helpers import many_matmul

def newton(f, grad_f, hess_f, x):
    print("""
    This doesn't work because the hessian is not always PD.
    There are ways around this (see 3.4 in NW). I just haven't implemented them.
    """)
    positions = [x]

    while np.linalg.norm(grad_f(x)) > 1e-7:
        B = hess_f(x)
        assert np.all(np.linalg.eig(B)[0] > 0)

        p = -np.matmul(np.linalg.inv(B), grad_f(x))
        p /= np.linalg.norm(p)

        a = get_line_length(f, grad_f, x, p, a_max=10)
        x = x + p * a
        positions.append(x)
    return np.array(positions)


def BFGS(f, grad_f, hess_0, x):
    dims = len(x)

    H = hess_0
    assert np.all(np.linalg.eig(H)[0] > 0) and np.all(H == H.T), "Initial hessian must be SPD"

    positions = [x]
    while np.linalg.norm(grad_f(x)) > 1e-7:
        p = -np.matmul(np.linalg.inv(hess_0), grad_f(x))
        # p /= np.linalg.norm(p)

        a = get_line_length(f, grad_f, x, p, a_max=10)

        x_next = x + a*p
        s = x_next - x
        y = grad_f(x_next) - grad_f(x)
        rho = 1 / np.matmul(y, s)

        # Note the second term is a scalar (rho) multiplied by a matrix
        t1 = (np.identity(dims) - rho * np.matmul(s[:,np.newaxis], y[np.newaxis,:]))
        t2 = (np.identity(dims) - rho * np.matmul(y[:,np.newaxis], s[np.newaxis,:]))
        t3 = rho * np.matmul(s[:,np.newaxis], s[np.newaxis,:])
        H = many_matmul(t1, H, t2) + t3

        x = x_next
        positions.append(x)
    return np.array(positions)

def L_BFGS(f, grad_f, x, memory_len=10):
    prev_s, prev_y, rho = [], [], []

    def estimate_search_dir(q, prev_s, prev_y, rho):
        assert len(prev_s) == len(prev_y) == len(rho)
        n = len(rho)

        # Just return the gradient first time around
        if n == 0:
            return q

        prev_s, prev_y, rho = np.array(prev_s), np.array(prev_y), np.array(rho)
        a = np.zeros_like(prev_s)

        for i in range(-1, -n-1, -1):
            a[i] = rho[i] * np.matmul(prev_s[i], q)
            q = q - np.matmul(a[i], prev_y[i])

        r = np.matmul(prev_s[-1], prev_y[-1]) / np.matmul(prev_y[-1], prev_y[-1]) * q

        for i in range(n):
            b = rho[i] * np.matmul(prev_y[i], r)
            r = r + prev_s[i] * (a[i] - b)

        return r


    positions = [x]
    i = 0
    # import pdb; pdb.set_trace()
    while np.linalg.norm(grad_f(x)) > 1e-2:
        if i % 1 == 0:
            print(x, np.linalg.norm(grad_f(x)))
        i += 1
        p = -estimate_search_dir(grad_f(x), prev_s, prev_y, rho)
        p /= np.linalg.norm(p)
        # print(np.linalg.norm(p))

        a = get_line_length(f, grad_f, x, p, a_max=1000, c1=1e-4, c2=0.80)

        x_next = x + a*p

        if len(prev_s) == memory_len:
            prev_s, prev_y, rho = prev_s[1:], prev_y[1:], rho[1:]
        prev_s.append(x_next - x)
        prev_y.append(grad_f(x_next) - grad_f(x))
        rho.append(np.matmul(prev_y[-1], prev_s[-1]))

        x = x_next
        positions.append(x)


    return np.array(positions)
