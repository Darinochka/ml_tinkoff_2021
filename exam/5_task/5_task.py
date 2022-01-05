import random
import numpy as np 

b = []

for i in range(1000):
    days = []

    for i in range(364):
        days.append(random.choice([0,1]))  # rain or not rain


    for i in range(len(days)):
        # if rain
        if days[i] == 1:
            # 2 - he forgot with 20% probability
            days[i] = random.choices([1, 2],weights = [0.8, 0.2])[0] 

    for i in range(len(days)):
        if days[i-1] == 2 and days[i] == 1:
            # probability of that the umbrella is not home
            days[i] = 5

    b.append(days.count(5))

print(np.mean(b)/364 * 100)
