import numpy as np
import itertools

class Minesweeper:
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nmines = 0
        self.flags = 0

        self.board_mines = np.array([])
        self.coord_mines = np.array([])
        self.opened_cell = set()

    def gen_coord_mines(self) -> list:
        """
        Generate random coordinate of
        locate mines
        """

        r = range(self.nrows)
        c = range(self.ncols)

        cells = np.array(np.meshgrid(r, c)).T.reshape(-1, 2)
        np.random.shuffle(cells)
        coord_bombs = cells[:self.nmines]
        
        return coord_bombs

    def init_board(self):
        """
        Generates board with mines with size (nrowx+2 x ncols+2).
        It is around zeros, so that it is easy to count mines around 
        any cell on real bord, where 0 - not mine 1 - mine
        and create hidden board for representation where * - mine
        """

        self.board_mines = np.zeros((self.nrows+2, self.ncols+2))
        self.hidden_board = [["*" for j in range(self.ncols)] for i in range(self.nrows)]
        self.flags = 0
        self.coord_mines = self.gen_coord_mines()

        # mark bomb cells
        for i, j in self.coord_mines:
            self.board_mines[i+1, j+1] = 1   

    def get_neighbour(self, cell: list) -> list:
        """
        Get coordinates of neighbors cross of this cell
        """
        y, x = cell
        neighbors = []

        for i, j in itertools.permutations([0, 1, -1], 2):
            if ({i, j} != {-1, 1} and 
                    0 <= y + i < self.nrows and 0 <= x + j < self.ncols):
                neighbors.append([y+i, x+j])
        return neighbors

    def open_surround_cells(self, user_coord: list, prev_cell: list):
        """
        Open cells which surrounding user cell, if
        they are 0 or 1
        """

        y = user_coord[0]
        x = user_coord[1]

        for i in self.get_neighbour([y, x]):
            if i != prev_cell and not self.is_mine(i):
                if (self.count_mines(i) == 0 or
                        self.count_mines(i) == 1):
                    # open this cell
                    self.hidden_board[i[0]][i[1]] = self.count_mines(i)
                    self.opened_cell.add(tuple(i))

                    if self.count_mines(i) == 0  and prev_cell == None: # deep of the recursion <= 2 
                        self.open_surround_cells(i, [y, x])
    
    def count_mines(self, cell: list):
        """
        Calculate the quantity of mines which
        surrounds cell
        """

        i, j = cell[0]+1, cell[1]+1     # board_mines size (ncrows+2 x ncols+2) -> +1
        return int(np.sum(self.board_mines[i-1:i+2, j-1:j+2]))

    def is_mine(self, cell: list):
        return cell in self.coord_mines.tolist()

    def update_hidden_table(self, user_coord: list, action: str = "open"):
        """
        Mark flag or open hidden board's cells
        based on the cell of gamer
        """

        y = user_coord[0]
        x = user_coord[1]

        if action == "flag":
            self.hidden_board[y][x] = "P"
            self.flags += 1

            self.beatiful_print_table(self.hidden_board)
            self.open_cell()
        else:
            if not self.is_mine([y, x]):
                self.opened_cell.add((y, x))
                
                self.hidden_board[y][x] = self.count_mines([y, x])
                self.open_surround_cells([y, x], None)

                if len(self.opened_cell) == self.ncols * self.nrows - self.nmines:
                    self.game_win()
                else:
                    self.beatiful_print_table(self.hidden_board)
                    self.open_cell()

            else:
                self.game_over()
                self.beatiful_print_table(self.hidden_board)

    def beatiful_print_table(self, table):
        print(f"Mines: {self.flags}/{self.nmines}")
        print("-"*5*self.ncols)
        for i in table:
            for j in i:
                print(f"| {j} |", end="")
            print()
            print("-"*5*self.ncols)
        print()

    def open_hidden_table(self):
        """
        Open cells of hidden table if user wins or loses
        """

        for i in range(self.nrows):
            for j in range(self.ncols):
                if [i, j] not in self.coord_mines.tolist():
                    self.hidden_board[i][j] = self.count_mines([i, j])
                else:
                    self.hidden_board[i][j] = "@"
