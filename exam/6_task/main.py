from game import GameUser
from ai import MinesweeperAI


def estimateAI():
    probabs = []
    for i in range(1000):
        gameAI = MinesweeperAI(5, 5, 2, False)
        gameAI.launch_game()
        result = gameAI.get_estimator()
        probabs.append(result)

    print(f"Percentage of wins: {probabs.count(1)/len(probabs)}")

def main():
    gameUser = GameUser()
    gameUser.launch_game()

if __name__ == "__main__":
    main()