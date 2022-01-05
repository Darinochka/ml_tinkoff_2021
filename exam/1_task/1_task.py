import itertools


def is_palindrom(tuple_num):
    for i in range(4):
        if tuple_num[i] != tuple_num[-(i+1)]:
            return False
    return True


def main():
    q = 0
    for i in itertools.product([0, 1], repeat=9):
        if is_palindrom(i):
            q += 1
            print(i)

    overall = 2**9
    print(q, overall)


if __name__ == "__main__":
    main()