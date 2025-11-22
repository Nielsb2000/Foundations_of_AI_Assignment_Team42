#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Evaluation functions for Competitive Sudoku AI.
Simple evaluation based on score difference.
"""

from competitive_sudoku.sudoku import GameState


class BoardEvaluator:
    """Handles board evaluation with simple score difference heuristic."""

    @staticmethod
    def evaluate_state(game_state: GameState, get_all_allowed_moves_func,
                      current_moves=None, opponent_moves_list=None) -> float:
        """
        Simple evaluation: score difference between current player and opponent.

        @param game_state: The current game state
        @param get_all_allowed_moves_func: Function to get all allowed moves (unused but kept for compatibility)
        @param current_moves: Pre-computed moves (unused but kept for compatibility)
        @param opponent_moves_list: Pre-computed opponent moves (unused but kept for compatibility)
        @return: Evaluation score (current_player_score - opponent_score)
        """
        current_player = game_state.current_player
        opponent = 3 - current_player

        current_score = game_state.scores[current_player - 1]
        opponent_score = game_state.scores[opponent - 1]

        return current_score - opponent_score
