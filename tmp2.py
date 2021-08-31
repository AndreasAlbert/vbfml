from matplotlib import pyplot as plt
import mplhep as hep
import numpy as np

nevent = 10000
fractions = [0.1,0.1,0.1,0.2,0.5]
values = np.random.dirichlet(fractions,size=nevent)
values = (values*nevent).astype(int)
ax = plt.gca()
for i in range(values.shape[-1]):
    print(i, np.sum(values[:,i]) / nevent, fractions[i])
    y, x = np.histogram(values[:,i])
    hep.histplot(y,x)
ax.figure.savefig("tmp2.pdf")