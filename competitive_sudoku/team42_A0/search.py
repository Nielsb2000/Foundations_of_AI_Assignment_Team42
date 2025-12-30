#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Search algorithms for Competitive Sudoku AI.
Contains Alpha-Beta pruning and other search strategies.
"""

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from team42_A0.zobrist_transposition_table import Zobrist


class AlphaBetaSearch:
    """Implements Alpha-Beta pruning for minimax search."""

    def __init__(self, evaluator, move_generator, use_tt: bool = True):
        """
        @param evaluator: BoardEvaluator instance for state evaluation
        @param move_generator: Function to generate all allowed moves
        """
        self.evaluator = evaluator
        self.move_generator = move_generator
        self.use_tt = use_tt
        self.tt = {}  # dict[ (hash, original_player) ] -> (depth, value)
        self.zobrist = None  # lazy init once we see a board

        self.tt_probes = 0
        self.tt_hits = 0
        self.tt_stores = 0

    @staticmethod
    def make_move(game_state: GameState, move: Move) -> None:
        """
        Applies a move to the game state (modifies in place).

        @param game_state: The game state to modify
        @param move: The move to apply
        """
        game_state.board.put(move.square, move.value)
        game_state.current_player = 3 - game_state.current_player

    @staticmethod
    def unmake_move(game_state: GameState, move: Move, previous_player: int) -> None:
        """
        Undoes a move on the game state (modifies in place).

        @param game_state: The game state to modify
        @param move: The move to undo
        @param previous_player: The player who made the move
        """
        game_state.board.put(move.square, SudokuBoard.empty)
        game_state.current_player = previous_player

    def _ensure_zobrist(self, board: SudokuBoard):
        if self.zobrist is None:
            self.zobrist = Zobrist(board=board, seed=0)  # deterministic seed for repeatability

    def clear_tt(self):
        """Clear the transposition table."""
        self.tt.clear()
        self.tt_probes = 0
        self.tt_hits = 0
        self.tt_stores = 0

    def tt_stats(self) -> dict:
        """Return statistics about the transposition table."""
        hit_rate = (self.tt_hits / self.tt_probes * 100) if self.tt_probes > 0 else 0
        return {
            'entries': len(self.tt),
            'probes': self.tt_probes,
            'hits': self.tt_hits,
            'hit_rate_percent': hit_rate,
            'stores': self.tt_stores,
            'memory_estimate_mb': len(self.tt) * 100 / (1024 * 1024)  # rough estimate
        }

    def print_tt_stats(self):
        """Print TT statistics to console."""
        stats = self.tt_stats()
        print(f"[TT Stats] Entries: {stats['entries']}, Probes: {stats['probes']}, "
              f"Hits: {stats['hits']}, Hit Rate: {stats['hit_rate_percent']:.1f}%, "
              f"Stores: {stats['stores']}")







    def alpha_beta(
        self,
        game_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        original_player: int,
        current_moves=None
    ) -> float:
        """
        Alpha-Beta pruning algorithm for minimax search.

        @param game_state: Current game state
        @param depth: Remaining search depth
        @param alpha: Best value for maximizing player
        @param beta: Best value for minimizing player
        @param maximizing_player: True if current player is maximizing
        @param original_player: The player we're evaluating for (stays constant)
        @param current_moves: Pre-computed moves for current state (optional, for optimization)
        @return: Best evaluation score FROM ORIGINAL PLAYER'S PERSPECTIVE
        """
        # Generate moves once if not provided
        if current_moves is None:
            current_moves = self.move_generator(game_state)

        # Transposition table probe (lazy init Zobrist)

        key = None
        if self.use_tt:
            self._ensure_zobrist(game_state.board)
            key = (self.zobrist.compute_hash(game_state), original_player)
            self.tt_probes += 1
            entry = self.tt.get(key)
            if entry is not None:
                stored_depth, stored_value = entry
                if stored_depth >= depth:
                    self.tt_hits += 1
                    return stored_value

        # Base case: reached depth limit or game over
        if depth == 0 or not current_moves:
            # Evaluate from original player's perspective
            eval_score = self.evaluator.evaluate_state(
                game_state, original_player, current_moves, self.move_generator
            )
            # Store in TT
            if self.use_tt and key is not None:
                self.tt[key] = (depth, eval_score)
                self.tt_stores += 1
            return eval_score

        if maximizing_player:
            max_eval = float('-inf')
            for move in current_moves:
                previous_player = game_state.current_player
                self.make_move(game_state, move)
                eval_score = self.alpha_beta(game_state, depth - 1, alpha, beta, False, original_player)
                self.unmake_move(game_state, move, previous_player)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            # Store in TT
            if self.use_tt and key is not None:
                self.tt[key] = (depth, max_eval)
                self.tt_stores += 1
            return max_eval
        else:
            min_eval = float('inf')
            for move in current_moves:
                previous_player = game_state.current_player
                self.make_move(game_state, move)
                eval_score = self.alpha_beta(game_state, depth - 1, alpha, beta, True, original_player)
                self.unmake_move(game_state, move, previous_player)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            # Store in TT
            if self.use_tt and key is not None:
                self.tt[key] = (depth, min_eval)
                self.tt_stores += 1
            return min_eval

    def find_best_move(self, game_state: GameState, depth: int) -> tuple[Move, float]:
        """
        Finds the best move using Alpha-Beta search.

        @param game_state: Current game state
        @param depth: Search depth
        @return: Tuple of (best_move, evaluation_score)
        """
        all_moves = self.move_generator(game_state)

        if not all_moves:
            return None, float('-inf')

        best_move = all_moves[0]
        best_evaluation = float('-inf')
        original_player = game_state.current_player

        for move in all_moves:
            previous_player = game_state.current_player
            self.make_move(game_state, move)
            evaluation = self.alpha_beta(
                game_state, depth - 1, float('-inf'), float('inf'), False, original_player
            )
            self.unmake_move(game_state, move, previous_player)

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move

        return best_move, best_evaluation
