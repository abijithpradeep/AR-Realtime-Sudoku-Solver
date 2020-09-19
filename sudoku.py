"""
Sudoku solver code.

Credits : https://norvig.com/sudoku.html
"""


import numpy as np


class Sudoku:
    def __init__(self, puzzle):
        self.sudoku = puzzle 
        self.digits = '123456789'
        self.rows = 'ABCDEFGHI'
        self.cols = self.digits
        self.squares = self.cross(self.rows, self.cols)
        self.unitlist = (
                          [self.cross(self.rows, c) for c in self.cols] +
                          [self.cross(r, self.cols) for r in self.rows] +
                          [self.cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
                        )
        self.units = dict(
                          (s, [u for u in self.unitlist if s in u])
                          for s in self.squares
                        )
        self.peers = dict(
                          (s, set(sum(self.units[s], [])) - set([s]))
                          for s in self.squares
                         )


    def cross(self, A, B):
        "Cross product of elements in A and elements in B."

        return [a + b for a in A for b in B]


    def grid_values(self, grid):
        "Convert grid into a dict of {square: char} with '0' or '.' for empties."

        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))


    def eliminate(self, values, s, d):
        """
        Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected.
        """

        if d not in values[s]:
            return values  

        values[s] = values[s].replace(d, '')
        if len(values[s]) == 0:
            return False  

        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self.eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False

        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False  
            elif len(dplaces) == 1:
                if not self.assign(values, dplaces[0], d):
                    return False

        return values


    def assign(self, values, s, d):
        """
        Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return False if a contradiction is detected.
        """

        other_values = values[s].replace(d, '')
        if all(self.eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return False


    def parse_grid(self, grid):
        """
        Convert grid to a dict of possible values, {square: digits}, or
        return False if a contradiction is detected.
        """

        values = dict((s, self.digits) for s in self.squares)
        for s, d in self.grid_values(grid).items():
            if d in self.digits and not self.assign(values, s, d):
                return False  

        return values


    def solve(self):
        tmp = self.search(self.parse_grid(self.sudoku))
        if tmp!=False:
            return np.array([tmp[s] for s in self.squares])
        else:
            return False


    def search(self, values):
        "Using depth-first search and propagation, try all possible values."

        if values is False:
            return False  
        if all(len(values[s]) == 1 for s in self.squares):
            return values 
        n, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)

        return self.some(self.search(self.assign(values.copy(), s, d)) for d in values[s])


    def some(self, seq):
        "Return some element of seq that is true."

        for e in seq:
            if e: 
                return e
        return False
