import board
import numpy as np
from board import TicTacToeEnv
from gymnasium.envs.registration import register
import gymnasium

ev = TicTacToeEnv(player=board.Player.WHITE, render_mode="human")
ev.reset()
breakpoint()
while True:
    ev.render()
