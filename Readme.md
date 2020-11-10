# AR Real-time Sudoku Solver
Sudoku is a logic based combinatorial number placement puzzle.
The goal of the puzzle is to fill 9x9 grid such that each of the nine blocks(3x3 grids) has to contain all the digits 1-9 and each number can only appear once in a row, column or box. [sudoku rules](https://www.sudokukingdom.com/rules.php)

This project make use of computer vision techniques to extract the sudoku from a live camera feed and displays the solution onto the actual frame after solving the puzzle in real time. It also crops out the latest solved sudoku image and saves it as "solvedSudoku.jpg".

<hr>

## Demo
<img src="output/demo.gif" width="512" height="420"/>

<hr>


## Requirements
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following.

```bash
pip install opencv-python
pip install tensorflow
pip install numpy
```

<hr>


## Usage
```python
python main.py
```

Press "Q" to quit the program.

## Credits
Thanks to [Peter Norvig](https://norvig.com/sudoku.html) for his amazing sudoku solver algorithm.