from multiprocessing import Pool
import random

def doubler(v, m):
    try:
        return _inner_doubler(v, m)
    except Exception as e:
        return 0

def _inner_doubler(v, m):
    if random.random() < 0.5:
        raise Exception("Blah")
    return v*m



p = Pool(5)

res = p.starmap(doubler, list(zip(range(10), range(10))))
print(res)
