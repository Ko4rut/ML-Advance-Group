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

## Connect 4 Task 3 — Minimax / Alpha-Beta (Tùng Thiện)

Code tìm kiếm nằm trong `connect4_minimax.py` (dùng lại `Position, empty_board, result, terminal, utility` từ `connect4_core.py`).
Notebook làm việc riêng của phần này là `connect4_task3_minimax.ipynb`; các mục của notebook khớp 1-1 với Task 3 trong `assignment_connect4.ipynb`.

Chạy từ thư mục này:

```bash
python -m unittest -v test_connect4_core test_connect4_minimax
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 connect4_task3_minimax.ipynb
```

Các hàm chính (mọi hàm tìm kiếm trả về dict `{"move", "value", "nodes"}`; `node_limit = None` là không giới hạn, vượt giới hạn thì `raise SearchLimitReached`; `verbose = 0` là không in gì, `1` in số nút, `2` in từng nút; không còn biến toàn cục `DEBUG`/`COUNT`):

* `minimax_search(board, player = 1, node_limit = None, verbose = 0)`
* `alpha_beta_search(board, player = 1, move_order = no_ordering, node_limit = None, verbose = 0)`
* Cả hai là một lớp `Search(...)` duy nhất (`prune = False` là Minimax, `prune = True` là Alpha-Beta); `Search.visit` thay cho bốn hàm `max_value`/`min_value`/`max_value_ab`/`min_value_ab` trước đây.
* `connect4_core.Position` — thế cờ đã kiểm tra một lần, nhớ người thắng; `drop(player, column)` trả về `Position` mới nên tìm kiếm không phải kiểm tra lại bàn cờ ở mỗi nút.
* Thứ tự nước đi (`move_order`): `no_ordering`, `center_first`, `win_block_center`. Một hàm sắp xếp nhận `(board, player, moves)` và trả về danh sách cột đã sắp xếp.
* `AlphaBetaAgent(move_order = ..., remember_moves = True).act(board, player = 1)` — tác tử dạng class, nhớ thế cờ đã tính (`known_moves`, `move_times`, `nodes`).
* `board_from_moves(shape, moves, first_player = 1)` — tạo bàn hợp lệ từ danh sách cột, trả về `(board, người_cần_đi)`.
* `timed_search(search, board, player = 1, **kwargs)` — chạy một hàm tìm kiếm và thêm `"time_ms"`, `"finished"` vào kết quả.
* `play(first, second, N = 100, shape = (4, 4))` — cho hai tác tử đấu N ván, trả về `{"first", "second", "draw"}`.

Gợi ý cho Heuristic Alpha-Beta (Task 4): muốn thêm cắt độ sâu có thể dựa theo `max_value_ab` / `min_value_ab`
giống cách `tictactoe_heuristic_alpha_beta_tree_search.ipynb` đã làm: thêm tham số `depth` / `cutoff` và hàm đánh giá vào `Search.visit` (chỉ một chỗ để sửa).

## Connect 4 Task 5 — Pure Monte Carlo (Toàn Thắng)

Code nằm trong `connect4_montecarlo.py` (dùng lại `Position`, `actions` từ `connect4_core.py`; `board_from_moves`, `AlphaBetaAgent`, `play`... từ `connect4_minimax.py` chỉ dùng trong test và notebook). Cấu trúc theo `tictactoe_pure_monte_carlo_search.ipynb`: người chơi là `1` / `-1`, ván đánh thử (playout) dùng `Position` cho nhanh.
Notebook làm việc riêng của phần này là `connect4_task5_montecarlo.ipynb`; các mục của notebook khớp 1-1 với Task 5 trong `assignment_connect4.ipynb`.

Chạy từ thư mục này:

```bash
python -m unittest -v test_connect4_core test_connect4_minimax test_connect4_montecarlo
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 connect4_task5_montecarlo.ipynb
```

Các hàm chính (biến toàn cục `DEBUG = 0` im lặng, `DEBUG = 1` in điểm trung bình của từng cột; mọi chỗ có ngẫu nhiên dùng `np.random`, nên đặt `np.random.seed(...)` trước để chạy lại ra đúng kết quả; seed của phần này là `20261110`):

* `playout(state, action, player = 1)` — thả quân vào `action` rồi cho hai bên đi ngẫu nhiên tới hết ván; trả về `+1` / `0` / `-1` nhìn từ `player`.
* `playouts(board, action, player = 1, N = 100)` — `N` ván đánh thử cho một cột.
* `pmcs_values(board, N = 100, player = 1)` — điểm trung bình của từng cột; `N` ván được chia đều cho các cột (mỗi cột ít nhất 1 ván).
* `pmcs(board, N = 100, player = 1)` — Pure Monte Carlo Search, trả về cột có điểm trung bình cao nhất (bằng điểm thì lấy cột bên trái). Bàn đã kết thúc thì `raise ValueError`.
* `PMCSAgent(N = 100).act(board, player = 1)` — tác tử dạng class, lưu thời gian mỗi nước (`move_times`) và số nước đã đi (`calls`).
* `pmcs10_player`, `pmcs100_player`, `pmcs1000_player` — tác tử dạng hàm, giống notebook Tic-Tac-Toe.

Tic-Tac-Toe: phần Bài 5 và Bài 6 của Toàn Thắng được **thêm ở cuối** hai notebook `tictactoe_pure_monte_carlo_search.ipynb` (Pure Monte Carlo) và `tictactoe_monte_carlo_tree_search_restricted.ipynb` (UCB1), cùng các ô nhận xét và ô đặt seed chèn giữa các ô gốc.

## Nộp bài Lab05

| File | Nội dung | Người phụ trách |
| --- | --- | --- |
| [tictactoe_definitions.ipynb](tictactoe_definitions.ipynb) | Bài 1: định nghĩa trò chơi, Random vs Random | Tú |
| [tictactoe_and_or_tree_search.ipynb](tictactoe_and_or_tree_search.ipynb) | Bài 2: AND-OR Tree Search | Tiến |
| [tictactoe_alpha_beta_tree_search.ipynb](tictactoe_alpha_beta_tree_search.ipynb) | Bài 3: Minimax và Alpha-Beta; câu thảo luận 2 | Thiện |
| [tictactoe_heuristic_alpha_beta_tree_search.ipynb](tictactoe_heuristic_alpha_beta_tree_search.ipynb) | Bài 4: Heuristic Alpha-Beta | Tiến |
| [tictactoe_pure_monte_carlo_search.ipynb](tictactoe_pure_monte_carlo_search.ipynb) | Bài 5: Pure Monte Carlo Search; câu thảo luận 4 | Thắng |
| [tictactoe_monte_carlo_tree_search_restricted.ipynb](tictactoe_monte_carlo_tree_search_restricted.ipynb) | Bài 6: UCB1; câu thảo luận 5 | Thắng |
| [tictactoe_interactive.ipynb](tictactoe_interactive.ipynb) | Bài 7: chơi tương tác | Tú |
| [assignment_connect4.ipynb](assignment_connect4.ipynb), `assignment_connect4.html` | Bài 8: Connect 4 (bản HTML là bản đã chạy để nộp) | Task 1–2: Tú; Task 3: Thiện; Task 4: Tiến; Task 5: Thắng |
| [connect4_core.py](connect4_core.py), [test_connect4_core.py](test_connect4_core.py) | Môi trường Connect 4 và test | Tú |
| [connect4_minimax.py](connect4_minimax.py), [test_connect4_minimax.py](test_connect4_minimax.py) | Minimax / Alpha-Beta (Task 3) | Thiện |
| [connect4_montecarlo.py](connect4_montecarlo.py), [test_connect4_montecarlo.py](test_connect4_montecarlo.py), [connect4_task5_montecarlo.ipynb](connect4_task5_montecarlo.ipynb) | Pure Monte Carlo (Task 5) và notebook làm việc | Thắng |
| `bao_cao.pdf`, `bao_cao.docx` | Báo cáo tổng hợp của nhóm | Thắng tổng hợp |

Chạy lại các phần Monte Carlo (từ thư mục này):

```bash
python -m unittest -v test_connect4_core test_connect4_minimax test_connect4_montecarlo
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 tictactoe_pure_monte_carlo_search.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 tictactoe_monte_carlo_tree_search_restricted.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 connect4_task5_montecarlo.ipynb
```

Notebook Connect 4 chung chạy lâu hơn (Task 3 có phần đo thời gian); sau khi chạy xong thì xuất lại bản HTML để nộp:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 assignment_connect4.ipynb
python -m jupyter nbconvert --to html assignment_connect4.ipynb
```

Thời gian trong các bảng đo trên máy của người chạy; các thí nghiệm có seed nên chạy lại ra đúng kết quả (thời gian thì khác theo máy). Báo cáo của nhóm nằm ở `bao_cao.pdf`.
