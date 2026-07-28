import numpy as np
from gsplatrender import add_one

a = np.ones((1000000,), dtype=np.float32)
b = add_one(a)

print(b)
