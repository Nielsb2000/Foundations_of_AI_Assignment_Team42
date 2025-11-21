#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Competitive Sudoku AI using Alpha-Beta pruning with enclosing strategy.
Team 42 - Assignment A0
"""

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

from team42_A0.evaluation import BoardEvaluator
from team42_A0.search import AlphaBetaSearch


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Competitive Sudoku AI that uses Alpha-Beta pruning for move selection.

    Strategy:
    - Uses minimax search with Alpha-Beta pruning (depth 4)
    - Evaluation focuses on score difference and restricting opponent mobility
    - Employs an "enclosing strategy" to limit opponent's options
    """

    def __init__(self):
        super().__init__()
        self.evaluator = BoardEvaluator()
        self.search = AlphaBetaSearch(self.evaluator, self.get_all_allowed_moves)
        self.search_depth = 3

    def get_all_allowed_moves(self, game_state: GameState) -> list[Move]:
        """
        Generates all allowed moves for the current game state.

        A move is allowed if:
        1. The cell is empty
        2. The move is not in taboo_moves (would make sudoku unsolvable)
        3. The cell is in player's allowed squares
        4. The value doesn't violate Sudoku rules

        @param game_state: The current game state
        @return: List of all allowed Move objects
        """
        N = game_state.board.N
        m = game_state.board.m
        n = game_state.board.n
        allowed_moves = []

        player_squares = game_state.player_squares()

        for i in range(N):
            for j in range(N):
                if game_state.board.get((i, j)) != SudokuBoard.empty:
                    continue

                if player_squares is not None and (i, j) not in player_squares:
                    continue

                for value in range(1, N + 1):
                    if TabooMove((i, j), value) in game_state.taboo_moves:
                        continue

                    if self._is_valid_move(game_state.board, i, j, value, m, n, N):
                        allowed_moves.append(Move((i, j), value))

        return allowed_moves

    def _is_valid_move(
        self,
        board: SudokuBoard,
        row: int,
        col: int,
        value: int,
        m: int,
        n: int,
        N: int
    ) -> bool:
        """
        Checks if placing a value at (row, col) violates Sudoku rules.

        @param board: The current Sudoku board
        @param row: Row index
        @param col: Column index
        @param value: Value to place (1 to N)
        @param m: Region height
        @param n: Region width
        @param N: Board size (m * n)
        @return: True if the move is valid, False otherwise
        """
        # Check row constraint
        for j in range(N):
            if board.get((row, j)) == value:
                return False

        # Check column constraint
        for i in range(N):
            if board.get((i, col)) == value:
                return False

        # Check region constraint
        region_row = (row // m) * m
        region_col = (col // n) * n
        for i in range(region_row, region_row + m):
            for j in range(region_col, region_col + n):
                if board.get((i, j)) == value:
                    return False

        return True

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes and proposes the best move using Alpha-Beta search.

        Uses an anytime algorithm approach:
        1. Immediately proposes a random valid move (safety)
        2. Searches for better moves using Alpha-Beta pruning
        3. Proposes improved moves as they are found
        4. Continues proposing the best move until time runs out

        @param game_state: The current game state
        """
        all_moves = self.get_all_allowed_moves(game_state)

        if not all_moves:
            return

        # Safety: immediately propose a random move
        best_move = random.choice(all_moves)
        self.propose_move(best_move)

        # Search for better moves using Alpha-Beta
        best_evaluation = float('-inf')
        original_player = game_state.current_player

        for move in all_moves:
            new_state = self.search.simulate_move(game_state, move)
            evaluation = self.search.alpha_beta(
                new_state,
                self.search_depth - 1,
                float('-inf'),
                float('inf'),
                False,
                original_player
            )

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move
                self.propose_move(best_move)

        # Continue proposing the best move found
        while True:
            self.propose_move(best_move)
