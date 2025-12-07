# Zobrist hashing utilities for the competitive_sudoku project.
# deterministic if seed provided (useful in tests)

import random
from typing import Optional

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard

class Zobrist:
    """
    Zobrist hashing for the two-player Sudoku game.
    - table[square_index][value]  : random uint64 for placing `value` at `square_index` (value in 1..N)
    - owner_table[square_index][0|1] : random uint64 for ownership by player1/player2
    - taboo_table[square_index][value] : random uint64 for a taboo move marking
    - side_to_move : random uint64 toggled when current_player == 2
    """

    def __init__(self, board: Optional[SudokuBoard] = None, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        N = board.N if board is not None else 9
        self.N = N
        self.num_squares = N * N
        self.values = N  # valid sudoku values are 1..N

        # table: num_squares x (values+1) (index 0 unused; values start at 1)
        self.table = [
            [self._rand64() for _ in range(self.values + 1)]
            for _ in range(self.num_squares)
        ]

        # owner table: two entries per square (player1, player2)
        self.owner_table = [
            [self._rand64(), self._rand64()] for _ in range(self.num_squares)
        ]

        # taboo table: like the piece table (for marking taboo moves)
        self.taboo_table = [
            [self._rand64() for _ in range(self.values + 1)]
            for _ in range(self.num_squares)
        ]

        # side to move bit
        self.side_to_move = self._rand64()

    def _rand64(self) -> int:
        return self._rng.getrandbits(64)

    def compute_hash(self, game_state: GameState) -> int:
        """
        Compute the Zobrist hash of a full GameState.
        Components:
         - board values (for each square with value v, xor table[k][v])
         - occupied_squares1/2 xor owner_table entries
         - taboo_moves xor taboo_table entries
         - current_player == 2 xor side_to_move
        """
        h = 0
        board = game_state.board

        # board values
        for k in range(board.N * board.N):
            val = board.squares[k]
            if val != SudokuBoard.empty:
                h ^= self.table[k][val]

        # ownership info (non-classic modes)
        if game_state.occupied_squares1:
            for sq in game_state.occupied_squares1:
                k = board.square2index(sq)
                h ^= self.owner_table[k][0]
        if game_state.occupied_squares2:
            for sq in game_state.occupied_squares2:
                k = board.square2index(sq)
                h ^= self.owner_table[k][1]

        # taboo moves
        if game_state.taboo_moves:
            for tm in game_state.taboo_moves:
                k = board.square2index(tm.square)
                h ^= self.taboo_table[k][tm.value]

        # side to move
        if game_state.current_player == 2:
            h ^= self.side_to_move

        return h

    # ---- Incremental helpers ----
    def xor_square_value(self, h: int, board: SudokuBoard, square: tuple, value: int) -> int:
        k = board.square2index(square)
        return h ^ self.table[k][value]

    def xor_owner(self, h: int, board: SudokuBoard, square: tuple, player_index: int) -> int:
        k = board.square2index(square)
        return h ^ self.owner_table[k][player_index]

    def xor_taboo(self, h: int, board: SudokuBoard, square: tuple, value: int) -> int:
        k = board.square2index(square)
        return h ^ self.taboo_table[k][value]

    def xor_side_to_move(self, h: int) -> int:
        return h ^ self.side_to_move

    # convenience high-level updates
    def apply_move(self, h: int, board: SudokuBoard, move: Move, player: int) -> int:
        """
        Update hash when applying a move (placing move.value at move.square by `player` 1 or 2).
        Assumes square was empty before applying. Returns new hash.
        """
        h = self.xor_square_value(h, board, move.square, move.value)
        idx = 0 if player == 1 else 1
        h = self.xor_owner(h, board, move.square, idx)
        h = self.xor_side_to_move(h)
        return h

    def undo_move(self, h: int, board: SudokuBoard, move: Move, player: int) -> int:
        """
        Undo the effects of apply_move. Returns hash after undoing.
        """
        h = self.xor_side_to_move(h)
        idx = 0 if player == 1 else 1
        h = self.xor_owner(h, board, move.square, idx)
        h = self.xor_square_value(h, board, move.square, move.value)
        return h