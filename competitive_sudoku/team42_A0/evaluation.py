#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Evaluation functions for Competitive Sudoku AI.
Evaluation based on score difference, central position control, and exclusive territory.
"""

from competitive_sudoku.sudoku import GameState, SudokuBoard, TabooMove
from typing import Optional, List, Tuple


class BoardEvaluator:
    """Handles board evaluation with score, central position control, and territory control."""

    @staticmethod
    def count_constraint_fills(board: SudokuBoard, square: tuple, value: int) -> int:
        """
        Counts how many constraints (row, column, region) would be completed
        if the given value is placed at the given square.

        @param board: The current board
        @param square: The square to check
        @param value: The value to place
        @return: Number of constraints that would be completed (0-3)
        """
        row, col = square
        N = board.N
        m = board.m
        n = board.n
        completed = 0

        # Check row completion
        row_full = True
        for j in range(N):
            if (row, j) == square:
                continue  # Skip the square we're evaluating
            if board.get((row, j)) == SudokuBoard.empty:
                row_full = False
                break
        if row_full:
            completed += 1

        # Check column completion
        col_full = True
        for i in range(N):
            if (i, col) == square:
                continue  # Skip the square we're evaluating
            if board.get((i, col)) == SudokuBoard.empty:
                col_full = False
                break
        if col_full:
            completed += 1

        # Check region completion
        region_row = (row // m) * m
        region_col = (col // n) * n
        region_full = True
        for i in range(region_row, region_row + m):
            for j in range(region_col, region_col + n):
                if (i, j) == square:
                    continue  # Skip the square we're evaluating
                if board.get((i, j)) == SudokuBoard.empty:
                    region_full = False
                    break
            if not region_full:
                break
        if region_full:
            completed += 1

        return completed

    @staticmethod
    def calculate_scoring_potential(game_state: GameState, player: int,
                                   move_generator_func) -> float:
        """
        Calculates the potential scoring value for a player based on moves
        that would complete constraints.

        Scoring: 1 point for 1 constraint, 3 for 2 constraints, 7 for 3 constraints

        @param game_state: The current game state
        @param player: The player to evaluate (1 or 2)
        @param move_generator_func: Function to generate valid moves
        @return: Weighted sum of potential scores
        """
        board = game_state.board
        N = board.N

        # Temporarily set current player to evaluate their moves
        original_player = game_state.current_player
        game_state.current_player = player

        # Generate all valid moves for this player
        valid_moves = move_generator_func(game_state)

        # Restore original player
        game_state.current_player = original_player

        potential_score = 0.0

        for move in valid_moves:
            constraints = BoardEvaluator.count_constraint_fills(board, move.square, move.value)

            # Convert constraints completed to actual points
            if constraints == 1:
                potential_score += 1
            elif constraints == 2:
                potential_score += 3
            elif constraints == 3:
                potential_score += 7

        return potential_score

    @staticmethod
    def count_exclusive_squares(game_state: GameState, player: int) -> int:
        """
        Counts empty squares that only the specified player can access.
        These are valuable because the opponent cannot contest them.

        @param game_state: The current game state
        @param player: The player to evaluate (1 or 2)
        @return: Number of exclusive empty squares for this player
        """
        # In classic mode, all squares are accessible to both players
        if game_state.is_classic_game():
            return 0

        board = game_state.board
        N = board.N
        opponent = 3 - player

        # Get allowed squares for each player
        player_allowed = game_state.allowed_squares1 if player == 1 else game_state.allowed_squares2
        opponent_allowed = game_state.allowed_squares1 if opponent == 1 else game_state.allowed_squares2

        if player_allowed is None:
            # Player can access all squares, so no exclusivity
            return 0

        exclusive_count = 0

        for i in range(N):
            for j in range(N):
                square = (i, j)

                # Skip if square is not empty
                if board.get(square) != SudokuBoard.empty:
                    continue

                # Check if player can access this square
                player_can_access = square in player_allowed

                # Check if opponent can access this square
                opponent_can_access = (opponent_allowed is None or square in opponent_allowed)

                # Exclusive if player can access but opponent cannot
                if player_can_access and not opponent_can_access:
                    exclusive_count += 1

        return exclusive_count

    @staticmethod
    def calculate_centrality_score(board: SudokuBoard, moves: list) -> float:
        """
        Calculates average centrality score for a list of moves.
        Central squares are more valuable strategically.

        @param board: The game board
        @param moves: List of Move objects
        @return: Average centrality score (higher is better)
        """
        if not moves:
            return 0.0

        N = board.N
        center = (N - 1) / 2.0  # Center coordinate

        total_centrality = 0.0

        for move in moves:
            i, j = move.square

            # Calculate distance from center (Manhattan distance)
            distance = abs(i - center) + abs(j - center)

            # Convert distance to a centrality score (closer = higher score)
            # Max distance is 2 * center, normalize to 0-1 range
            max_distance = 2 * center
            centrality_value = 1.0 - (distance / max_distance)

            total_centrality += centrality_value

        return total_centrality / len(moves)

    @staticmethod
    def evaluate_state(game_state: GameState, original_player: int,
                      current_moves=None, move_generator_func=None) -> float:
        """
        Evaluation from the original player's perspective.

        Components:
        1. Score difference (weight: 10.0) - actual points earned
        2. Scoring potential difference (weight: 1.0) - immediate scoring opportunities
        3. Exclusive territory control (weight: 2.5) - squares only this player can access
        4. Central position control (weight: 0.4) - preference for central squares

        @param game_state: The current game state
        @param original_player: The player we are evaluating for (1 or 2)
        @param current_moves: Pre-computed moves for current player
        @param move_generator_func: Function to generate valid moves (needed for scoring potential and centrality)
        @return: Evaluation score from original_player's perspective
        """
        # Get scores from original player's perspective
        original_score = game_state.scores[original_player - 1]
        opponent = 3 - original_player
        opponent_score = game_state.scores[opponent - 1]

        score_diff = original_score - opponent_score

        # Territory control: exclusive squares for each player
        original_exclusive = BoardEvaluator.count_exclusive_squares(game_state, original_player)
        opponent_exclusive = BoardEvaluator.count_exclusive_squares(game_state, opponent)
        territory_diff = original_exclusive - opponent_exclusive

        # Scoring potential: sum of potential points from constraint completions
        scoring_potential_diff = 0.0
        if move_generator_func is not None:
            original_potential = BoardEvaluator.calculate_scoring_potential(
                game_state, original_player, move_generator_func
            )
            opponent_potential = BoardEvaluator.calculate_scoring_potential(
                game_state, opponent, move_generator_func
            )
            scoring_potential_diff = original_potential - opponent_potential

        # Central position control: average centrality of available moves
        centrality_diff = 0.0
        if move_generator_func is not None:
            # Generate moves for both players
            original_game = game_state
            original_game.current_player = original_player
            original_moves = move_generator_func(original_game)
            original_centrality = BoardEvaluator.calculate_centrality_score(
                game_state.board, original_moves
            )

            original_game.current_player = opponent
            opponent_moves = move_generator_func(original_game)
            opponent_centrality = BoardEvaluator.calculate_centrality_score(
                game_state.board, opponent_moves
            )

            # Restore original current player
            original_game.current_player = game_state.current_player

            centrality_diff = original_centrality - opponent_centrality

        # Combine all factors with weights
        # Score is most important (10x), then scoring potential (1x), territory (2.5x), centrality (0.4x)
        evaluation = (
            (score_diff * 10.0) +
            (scoring_potential_diff * 1.0) +
            (territory_diff * 2.5) +
            (centrality_diff * 0.4)
        )

        return evaluation 