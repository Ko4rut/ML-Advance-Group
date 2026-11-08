# Adversarial Search and Games

## Running the Tic-Tac-Toe and Connect 4 work

Run these commands from this directory so notebook imports resolve to the local
`tictactoe.py` and `connect4_core.py` files:

```powershell
python -m pip install -r "..\..\requirements.txt"
python -m unittest -v test_connect4_core
jupyter lab tictactoe_definitions.ipynb tictactoe_interactive.ipynb assignment_connect4.ipynb
```

In the interactive Tic-Tac-Toe notebook, set `PLAY_AS_X` or `PLAY_AS_O` to
`True` in the corresponding cell to start a live game. Enter `-1` to quit.
Connect 4's Task 1–2 implementation is in `connect4_core.py`; its seeded
1,000-game Random-vs-Random experiment and results are in
`assignment_connect4.ipynb`.

To export the Connect 4 notebook after its saved outputs are in place:

```powershell
jupyter nbconvert --to html assignment_connect4.ipynb
```

## Defining a Game
* [Defining Tic-Tac-Toe](tictactoe_definitions.ipynb)

## Connection to search with nondeterministic actions
* [Solving Tic-Tac-Toe with And-Or-Tree Search](tictactoe_and_or_tree_search.ipynb). Here the opponent is seen as part of the environment, i.e.,
each action by the player is followed by an unknown action of the opponent which, from the viewpoint of the player makes the outcomes of actions nondeterministic.
* Here are the basic functions for 
[Tic-Tac-Toe](tictactoe_definitions.ipynb).

## Search in Unknown Environments
* Experimental environment: [Micromouse](https://en.wikipedia.org/wiki/Micromouse) is an event where small robotic mice compete to solve a maze. 
You can watch the video [The Fastest Maze-Solving Competition On Earth](https://www.youtube.com/watch?v=ZMQbHMgK2rw).
You can test your own agent function using the 
  [Micromouse Simulator](https://github.com/mackorone/mms-python). 

## Solving Games Using Adversarial Search
* [Solving Tic-Tac-Toe with Minimax Search and Alpha-Beta Pruning](tictactoe_alpha_beta_tree_search.ipynb)
* [Solving Tic-Tac-Toe with Heuristic Alpha-Beta Tree Search](tictactoe_heuristic_alpha_beta_tree_search.ipynb)
* [Solving Tic-Tac-Toe with Pure Monte Carlo Search](tictactoe_pure_monte_carlo_search.ipynb)
* [Solving Tic-Tac-Toe with Pure Monte Carlo Search + UCB1 Selection Policy](tictactoe_monte_carlo_tree_search_restricted.ipynb)

## Interactive Game Play
* [Play Tic-Tac-Toe Interactively (Simple Implementation)](tictactoe_interactive.ipynb)

## Exercises
* [Adversarial Search: Playing Connect 4](assignment_connect4.ipynb)
* [Adversarial Search: Playing "Mean" Connect 4](assignment_mean_connect4.ipynb)
* [Adversarial Search: Playing Dots and Boxes](assignment_dots_and_boxes.ipynb)