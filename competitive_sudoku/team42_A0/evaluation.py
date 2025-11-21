#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Evaluation functions for Competitive Sudoku AI.
Contains heuristics for evaluating game states and board positions.
"""

import copy
from competitive_sudoku.sudoku import GameState, SudokuBoard


class BoardEvaluator:
    """Handles all board evaluation and heuristic calculations."""

    @staticmethod
    def analyze_completable_structures(game_state: GameState, get_all_allowed_moves_func) -> dict:
        """
        Analyzes completable structures and determines which player can actually complete them.

        @param game_state: The current game state
        @param get_all_allowed_moves_func: Function to get all allowed moves
        @return: Dictionary with 'our_completable' and 'opponent_completable' counts
        """
        board = game_state.board
        N = board.N
        current_player = game_state.current_player
        opponent = 3 - current_player

        # Get moves for current player
        our_moves = get_all_allowed_moves_func(game_state)
        our_move_squares = {move.square for move in our_moves}

        # Get moves for opponent
        opponent_state = copy.deepcopy(game_state)
        opponent_state.current_player = opponent
        opponent_moves = get_all_allowed_moves_func(opponent_state)
        opponent_move_squares = {move.square for move in opponent_moves}

        our_completable = 0
        opponent_completable = 0

        # Check rows
        for row in range(N):
            if BoardEvaluator.count_empty_cells_in_row(board, row) == 1:
                # Find the empty cell
                for col in range(N):
                    if board.get((row, col)) == 0:  # Empty cell
                        if (row, col) in our_move_squares:
                            our_completable += 1
                        if (row, col) in opponent_move_squares:
                            opponent_completable += 1
                        break

        # Check columns
        for col in range(N):
            if BoardEvaluator.count_empty_cells_in_column(board, col) == 1:
                # Find the empty cell
                for row in range(N):
                    if board.get((row, col)) == 0:  # Empty cell
                        if (row, col) in our_move_squares:
                            our_completable += 1
                        if (row, col) in opponent_move_squares:
                            opponent_completable += 1
                        break

        return {
            'our_completable': our_completable,
            'opponent_completable': opponent_completable
        }

    @staticmethod
    def evaluate_state(game_state: GameState, get_all_allowed_moves_func) -> float:
        """
        Evaluates the current game state from the perspective of the current player.
        Higher scores indicate better positions for the current player.

        Strategy:
        1. Score difference between players (primary objective)
        2. Board dominance (number of available moves)
        3. Completable structures - differentiated by who can complete them
        4. Opponent restriction - reducing opponent's moves (balanced lockout)

        @param game_state: The current game state
        @param get_all_allowed_moves_func: Function to get all allowed moves for a state
        @return: Evaluation score (higher = better for current player)
        """
        current_player = game_state.current_player
        opponent = 3 - current_player

        # Factor 1: Score difference (most important)
        current_score = game_state.scores[current_player - 1]
        opponent_score = game_state.scores[opponent - 1]
        score_difference = current_score - opponent_score

        # Factor 2: Board dominance (mobility)
        current_player_moves = len(get_all_allowed_moves_func(game_state))

        opponent_state = copy.deepcopy(game_state)
        opponent_state.current_player = opponent
        opponent_moves = len(get_all_allowed_moves_func(opponent_state))

        # Factor 3: Completable structures (who can actually complete them)
        structure_analysis = BoardEvaluator.analyze_completable_structures(
            game_state, get_all_allowed_moves_func
        )

        # Reward structures we can complete, penalize structures opponent can complete
        structure_score = (
            structure_analysis['our_completable'] * 15 -      # We can score here
            structure_analysis['opponent_completable'] * 20   # Opponent threatens to score
        )

        # Factor 4: Opponent restriction (balanced lockout strategy)
        if opponent_moves == 0:
            # Win condition: opponent has no moves
            lockout_bonus = 500
        elif opponent_moves <= 2:
            # Very restricted opponent
            lockout_bonus = 100
        elif opponent_moves <= 5:
            # Moderately restricted opponent
            lockout_bonus = 50
        else:
            # Penalize opponent having many moves
            lockout_bonus = -opponent_moves * 3

        # Combine all factors with weights
        evaluation = (
            score_difference * 100 +        # Actual score (most important)
            current_player_moves * 2 +      # Our mobility
            structure_score +               # Threat assessment & opportunities
            lockout_bonus                   # Opponent restriction
        )

        return evaluation

    @staticmethod
    def count_empty_cells_in_row(board: SudokuBoard, row: int) -> int:
        """Counts the number of empty cells in a given row."""
        N = board.N
        empty_count = 0
        for j in range(N):
            if board.get((row, j)) == SudokuBoard.empty:
                empty_count += 1
        return empty_count

    @staticmethod
    def count_empty_cells_in_column(board: SudokuBoard, col: int) -> int:
        """Counts the number of empty cells in a given column."""
        N = board.N
        empty_count = 0
        for i in range(N):
            if board.get((i, col)) == SudokuBoard.empty:
                empty_count += 1
        return empty_count

    @staticmethod
    def count_empty_cells_in_region(board: SudokuBoard, row: int, col: int) -> int:
        """Counts the number of empty cells in the region containing (row, col)."""
        m = board.m
        n = board.n
        region_row = (row // m) * m
        region_col = (col // n) * n
        empty_count = 0
        for i in range(region_row, region_row + m):
            for j in range(region_col, region_col + n):
                if board.get((i, j)) == SudokuBoard.empty:
                    empty_count += 1
        return empty_count
