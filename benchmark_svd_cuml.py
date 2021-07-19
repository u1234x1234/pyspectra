import time

import numpy as np
from cuml.decomposition import TruncatedSVD

N_COMP = 64

X = np.random.normal(0, 20, size=(20_000, 4096)).astype(np.float32)

model = TruncatedSVD(algorithm="jacobi", n_components=N_COMP)
X_transformed = model.fit_transform(X)


model = TruncatedSVD(algorithm="jacobi", n_components=N_COMP)
st = time.time()
model.fit(X)
print(time.time() - st)
