import numpy as np
from sklearn.preprocessing import scale
def my_scale(feature_map):
    x = np.array(feature_map)
    shape = x.shape
    x = x.reshape([1,shape[0],shape[1]*shape[2]])
    x = scale(x[0])
    x = x.reshape(shape)
    return x


