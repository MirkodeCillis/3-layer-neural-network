# Executes the Backward query of the neural network given

import matplotlib.pyplot as plt
from guessNumberNetwork import ngn
import numpy as np

examine = 3
out_list = np.zeros(ngn.onodes) + 0.01
out_list[examine] = 0.99
backward_out = ngn.backquery(out_list)
image_data = backward_out.reshape((28, 28))
plt.imshow(image_data, cmap="Greys")
