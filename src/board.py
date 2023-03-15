from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as mpl
import math
import keras.utils as keras_utils
from typing import Tuple, Optional, Literal
import pygame

import keras.layers as layers

import gymnasium as gym
from gymnasium import spaces

WINNING_MULT = 1

class Player(Enum):
    BLACK = 1
    WHITE = 2

    def switch(self) -> "Player":
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK


class BoardState(Enum):
    DRAW = 0
    DONE = 1
    ONGOING = 2


class Board:

    def __init__(self) -> None:
        self.board = [0] * 9

    @classmethod
    def from_board(cls, board: list[int]) -> "Board":
        b = Board()
        b.board = board.copy()
        return b
    
    def copy(self) -> "Board":
        return Board.from_board(self.state)

    def index(self, x: int, y: int) -> int:
        return np.ravel_multi_index((x, y), (3, 3))[0]

    def done(self) -> bool:
        return np.all(np.array(self.board) != 0)

    def move(self, pl: Player, x: int) -> Tuple[int, list[int], bool]:
        valid = self.perform_move(pl, x)
        state = self.get_state()
        if valid and state == BoardState.ONGOING:
            return 0, self.board,  False
        if valid and state == BoardState.DONE:
            winner = self.get_winner()
            return (WINNING_MULT if winner == pl else -WINNING_MULT)  , self.board, True
        if valid and state == BoardState.DRAW:
            return 0, self.board, True
        if not valid and state == BoardState.ONGOING:
            return -WINNING_MULT, self.board, True

    
    def perform_move(self, pl: Player, x:int) -> bool:
        occupied = (self.board[x] != 0)
        if not occupied:
            self.board[x] = pl.value
            return True
        else:
            return False





    def to_np(self) -> np.array:
        return np.reshape(self.board, (3, 3))

    def to_np_flat(self) -> np.array:
        return np.array(self.board)

    def state(self) -> np.array:
        return self.to_np()

    def from_ks_input(self, x: np.array) -> Tuple[int, int]:
        return np.argmax(x, axis=-1)

    def show(self, ax: mpl.Axes):
        ax.imshow(self.to_np())

    def copy(self) -> "Board":
        return Board.from_board(self.board)

    def possible_moves(self, pl: Player) -> list[int]:
        return [i for i, j in enumerate(self.board) if j == 0]

    def winning(self, pl: Player) -> int:
        board_array = self.to_np()
        rows = np.sum(board_array == pl.value, axis = 0)
        cols = np.sum(board_array == pl.value, axis = 1)
        diag = np.trace(board_array == pl.value)
        anti_diag = np.trace(np.fliplr(board_array) == pl.value)
        def who(x): return np.any(x == (3))
        return who(rows) or who(cols) or who(diag) or who(anti_diag)

    def get_state(self) -> BoardState:
        done = self.done()
        winner = self.get_winner()
        if done and winner is None:
            return BoardState.DRAW
        if winner is not None:
            return BoardState.DONE
        else:
            return BoardState.ONGOING

    def get_winner(self) -> Optional[Player]:
        if self.winning(Player.WHITE):
            return Player.WHITE
        if self.winning(Player.BLACK):
            return Player.BLACK
        else:
            return None


NUM_LOC = 9


def get_color(pl: Player):
    match pl:
        case Player.WHITE:
            return (200, 200, 200)
        case Player.BLACK:
            return (0, 0, 0)


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, player:Player = Player.WHITE, render_mode: Optional[Literal["human", "rgb_array"]] = "rgb_array") -> None:
        self.render_mode = render_mode
        self.size = 3  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.observation_space = spaces.Discrete(NUM_LOC * 2) 
        self.action_space = spaces.Discrete(NUM_LOC)
        self.player = player
        self.window = None
        self.clock = None
        self.turn = self.player
        self.encoder = layers.CategoryEncoding(num_tokens=3, output_mode="one_hot")


    def _observation(self):
        board = self.board.to_np_flat()
        out_idx = [pos + (i == 2) * (NUM_LOC - 1) for pos, i in enumerate(board) if i != 0]
        out_arr = np.zeros(NUM_LOC * 2)
        out_arr[out_idx] = 1
        return out_arr
        

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.board = Board()
        return self._observation(), {}

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        score, state, done = self.board.move(self.turn, action)
        self.turn = self.turn.switch()
        
        return  self._observation() , score, done, {}
    
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for pos, el in enumerate(self.board.board):
            x_idx, y_idx = np.unravel_index(pos, (self.size, self.size))
            if el != 0:
                pygame.draw.rect(
                    canvas,
                    get_color(Player(el)),
                    pygame.Rect(
                        (pix_square_size * x_idx, pix_square_size * y_idx),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

