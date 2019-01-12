# The func below might be better
def finite_diff(f, *args):
    eps = 1
    while eps > 1e-30:
        diff = (
                f(*[args[i] + eps * (i == 0) for i in range(len(args))]) -
                f(*[args[i] - eps * (i == 0) for i in range(len(args))])
        )

        if diff < 1e-5:
            return diff/(2*eps)

        eps /= 2

    raise Exception("No converge! Function varies too rapidly.")

# import math

# # See Numerical Optimization Nocedal and Wright chapter 8 or
# # https://math.stackexchange.com/questions/815113/is-there-a-general-formula-for-estimating-the-step-size-h-in-numerical-different
# def finite_diff(f, *args):
#     ulp = 1e-53
#     eps = args[0] * math.sqrt(ulp)
#     diff = (
#             f(*[args[i] + eps * (i == 0) for i in range(len(args))]) -
#             f(*[args[i] - eps * (i == 0) for i in range(len(args))])
#     )

#     return diff/(2*eps)

print(finite_diff(lambda x1: x1**3, 2))
