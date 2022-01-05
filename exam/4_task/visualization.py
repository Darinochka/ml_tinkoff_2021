import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

c = 13/24
a = []
b = []
q = 0

for i in range(1000):
    a_temp = np.array([random.uniform(0, 2/3) for i in range(100)]).mean()
    b_temp = np.array([random.uniform(1/2, 1) for i in range(100)]).mean()

    a.append(a_temp)
    b.append(b_temp)

    if a[i] < c < b[i]:
        q += 1

fig, ax = plt.subplots(figsize=(15, 8))

ax = sns.distplot([a, b], bins=100, kde=True)
plt.xticks(np.linspace(0, 1, 25))
plt.show()
print(q)