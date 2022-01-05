from minesweeper import Minesweeper
import numpy as np
import random
import itertools


class MinesweeperAI(Minesweeper):
    def __init__(self, nrows, ncols, nmins, print_board=True):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.nmines = nmins
        self.print_board = print_board

        self.result = 0
        r = range(self.nrows)
        c = range(self.ncols)

        self.prob_board = np.zeros((self.nrows, self.ncols))
        self.closed_cells = np.array(np.meshgrid(r, c)).T.reshape(-1, 2).tolist()

    def launch_game(self):
        if self.print_board:
            print("Game start!")

        self.init_board()

        self.beatiful_print_table(self.hidden_board)
        self.open_cell(True)

    def get_neighbour(self, cell: list):
        """
        Overrides the parent function to get the coordinates 
        of the cell's neighbors not only with a cross.
        This is necessary to calculate the probability of being a mine 
        for each neighboring cell.
        """
        y, x = cell
        neighbors = []

        for i, j in itertools.product([0, 1, -1], repeat=2):
            if (0 <= y + i < self.nrows and 0 <= x + j < self.ncols):
                    neighbors.append([y+i, x+j])
        return neighbors

    def calc_prob_be_mine(self, cell):
        """
        Calculates the probability for the neighbors of 
        a given cell to be a mine. If the current cell is zero, 
        it takes away 1 for all neighboring cells.
        """
        if cell not in self.coord_mines.tolist():
            count_mines = self.count_mines(cell)
            neighbors = self.get_neighbour(cell)
            for i in neighbors:
                y = i[0]
                x = i[1] 
                if count_mines == 0:
                    self.prob_board[y][x] = -1
                    self.delete_cell(i)
                elif i in self.closed_cells:
                    self.prob_board[y, x] += count_mines/len(neighbors)

    def open_cell(self, first=False): # first move?
        """
        Opens a cell if the number of remaining cells is not equal 
        to the coordinates of the bombs. Selects the cell with t
        he lowest probability or randomly, if this is the first move.
        For an open cell, sets the probability -1.
        """
        self.calc_prob_open_cells()

        if self.nmines == len(self.closed_cells):
            if ((np.sort(self.coord_mines, axis=0) == \
                    np.sort(self.closed_cells, axis=0)).all()):
                self.game_win()
            else:
                self.game_over()
        else:
            if first:
                cell = self.random_move(self.closed_cells)
            else:
                cell = self.choose_cell()

            self.calc_prob_be_mine(cell)
            self.prob_board[cell[0], cell[1]] = -1  # be mine - 0% probability
            self.calc_prob_open_cells()
            self.update_hidden_table(cell)

    def calc_prob_open_cells(self):
        """
        Calculates probabilities for open cells
        """
        for cell in list(self.opened_cell):
            self.calc_prob_be_mine(cell)
            self.prob_board[cell[0], cell[1]] = -1
            self.delete_cell(cell)

    def choose_cell(self) -> list:
        """
        Selects the cells with the lowest probability from 
        the sorted probability board. Next, he randomly selects 
        one of all of them. If they are already open, 
        it goes further along the sorted sheep.
        """
        temp = list(set(self.prob_board.reshape(self.ncols*self.nrows)))
        idx = 0
        i = 0
        while not idx:
            k, j = np.where(self.prob_board == temp[i])
            coord = list(zip(k, j))
            idx = self.random_move(coord)
            i += 1
        return idx

    def random_move(self, arr: list) -> list:
        move = list(random.choice(arr))

        if move in self.closed_cells:
            self.delete_cell(move)
            return move
        else:
            return []

    def delete_cell(self, cell):
        """
        Removes a cell from the available closed cells, since it is open.
        """
        if cell in self.closed_cells:
            self.closed_cells.remove(cell)

    def game_win(self):
        if self.print_board:
            print("I win!")
        self.open_hidden_table()
        self.beatiful_print_table(self.hidden_board)
        self.result = 1

    def game_over(self):
        if self.print_board:
            print("Sorry, my creator :(")
        self.open_hidden_table()
        self.result = 0

    def get_estimator(self):
        return self.result

    def beatiful_print_table(self, table):
        if self.print_board:
            print(f"Mines: {self.flags}/{self.nmines}")
            print("-"*5*self.ncols)
            for i in table:
                for j in i:
                    print(f"| {j} |", end="")
                print()
                print("-"*5*self.ncols)
            print()