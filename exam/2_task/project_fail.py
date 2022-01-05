from math import comb, factorial as f


combinations = lambda n, k: f(n) / (f(k) * f(n - k))

fail = 0.1
bankrot = [0.2, 0.5, 0.7, 0.9]

result = 0
for i in range(1, 5):
    result += bankrot[i-1] * (fail) ** i * (1 - fail) ** (4 - i) * combinations(4, i)
print(result)