import random
import csv
from minesweeper import Minesweeper
from ai import MinesweeperAI

class GameUser(Minesweeper):
    def __init__(self):
        super().__init__()
        self.num_save = 0
        self.saving_games = []

    def show_history(self) -> int:
        """
        Show history of games in this session
        and gets game from user
        """

        for i, item in enumerate((self.saving_games)):
            print(f"{i}) size: {item['rows']}:{item['columns']}"
                f" number of the bombs: {len(self.coord_mines)}")

        num_game = int(input())
        return num_game

    def select_mode(self):
        """
        User chooses the mode of game:
        Base or his option for game
        """

        level = input()

        if level == "Base" or level == "1":
            self.nrows = 5
            self.ncols = 5
            self.nmines = random.randint(2, 5)

        elif level == "Optional" or level == "2":
            self.nrows = int(input("Please, enter the number of rows: "))
            self.ncols = int(input("Please, enter the number of columns: "))
            self.nmines = int(input("Please, enter the number of bombs: "))

            while self.nmines == 0 or self.nmines >= self.nrows * self.ncols:
                if self.nmines == 0:
                    print("The number of bombs cannot be equal to zero!")
                elif self.nmines >= self.nrows * self.ncols:
                    print("The number of bombs cannot be equal to number of cells")
                else:
                    break
                self.nmines = int(input("Please, enter the number of bombs: "))
        
                
        else:
            print("Wrong command! Try again...")
            self.select_mode()

    def launch_game(self):
        print("----------------1. New game----------------")
        print("----------------2. Continue----------------")
        print("----------------3. Robot----------------")
        
        decision = input()

        if decision == "1" or decision.lower() == "new game":
            print("""---------Choose the level of game----------
            1. Base (5x5, 2-5 bombs)
            2. Optional (your settings)""")

            self.select_mode()
            self.init_board()

            self.beatiful_print_table(self.hidden_board)
            self.open_cell()

        elif decision == "2" or decision.lower() == "continue":
            if self.saving_games:
                print("Which one?")
                num_game = self.show_history()

                if num_game in range(len(self.saving_games)):
                    self.load_game(num_game)
            else:
                print("History is empty!")
                self.launch_game()

        elif decision == "3" or decision.lower() == "robot":
            gameAI = MinesweeperAI(5, 5, 2)
            gameAI.launch_game()
            self.offer_game()
        else:
            print("Wrong command! Try again...")
            self.launch_game()

    def load_game(self, n_game: int):
        """
        Load the game with hidden cells from
        current folder in format {num}game.csv and
        inisializates setting of the game
        """
        
        print(self.saving_games[n_game])
        self.nrows = self.saving_games[n_game]["rows"]
        self.ncols = self.saving_games[n_game]["columns"]

        self.coord_mines = self.saving_games[n_game]["coord_bombs"]
        
        self.hidden_board = []
        with open(f"{n_game}game.csv") as f:
            file_reader = csv.reader(f, delimiter = ",")
            for row in file_reader:
                self.hidden_board.append([c for c in row][:-1])

        self.beatiful_print_table(self.hidden_board)
        self.open_cell()

    def open_cell(self):
        """
        Gets user's cell for opened. 
        Also, open cells arount it if they 0 or 1. If cell - mine, 
        the game is finished. And if all of cells is opened
        user wins
        """

        saving_either = input("Do you want to save this step of the game? y\\n: ")
        if saving_either == "y":
            self.save_game()
            print("This game is saved!")

        try:
            print("Please, enter the number of row and column: ")
            user_row = int(input())
            user_column = int(input())

            if (0 <= user_row < self.nrows and       # [0, y-1]
                    0 <= user_column < self.ncols):  # [0, x-1]

                print("Please, enter the action: 'flag' or 'open'")
                action = input()

                while action != "flag" and action != "open":
                    print("Please, enter the action: 'flag' or 'open'")
                    action = input()

                self.update_hidden_table([user_row, user_column], action)

            else:
                print("Wrong input data! Try again...")
                self.open_cell()
    
        except ValueError:
            print("Wrong input data! Try again...")
            self.open_cell()

    def game_over(self):
        print("-"*31)
        print("-"*10, "GAME OVER", "-"*10)
        print("-"*31)

        self.open_hidden_table()
        self.beatiful_print_table(self.hidden_board)

        self.offer_game()

    def offer_game(self):
        desiciion = input("Do you want to play again? y\\n: ")
        if desiciion == "y":
            self.launch_game()
        else:
            exit()

    def game_win(self):
        print("-"*31)
        print("-"*10, "YOU WIN!", "-"*10)
        print("-"*31)
        self.open_hidden_table()
        self.beatiful_print_table(self.hidden_board)

        self.offer_game()
        
    def save_game(self):
        """
        Saves file "{num}game.csv" in current folder
        and adds the current game in list of the saving games 
        """

        with open(f"{self.num_save}game.csv", "w") as f:
            for item in self.hidden_board:
                for i in item:
                    f.write(f"{i},")
                f.write("\n")

        save_game = {   
            "rows": self.nrows,
            "columns": self.ncols,
            "coord_bombs": self.coord_mines,
        }

        self.saving_games.append(save_game)
        self.num_save += 1
