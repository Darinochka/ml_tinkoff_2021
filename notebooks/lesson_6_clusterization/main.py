n = int(input())
columns = list(map(int, input().split()))

while len(set(columns)) != 1: # пока не сравняются

    # когда максимальный и минимальный элементы имеют разницу 1,
    # то их уравнять нельзя
    if max(columns) - min(columns) == 1:
        print('NO')
        break

    columns = sorted(columns)
    for i in range(n-1):
        columns[i] += 1
    columns[-1] -= 1

else: # этот блок не сработает, если будет вызван break
    print('YES')