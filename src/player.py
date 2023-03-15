
import board
import numpy as np
import math
from typing import Optional


def minimax(maxTurn: bool, player: board.Player, current_board: board.Board):
    state = current_board.get_state()
    if (state == board.BoardState.DRAW):
        return 0
    elif (state == board.BoardState.DONE):
        return 1 if current_board.get_winner() == player else -1

       

    scores = []
    for move in current_board.possible_moves(player):
        next_board = current_board.copy()
        next_board.move(player, move)
        scores.append(minimax(not maxTurn, player.switch(), next_board))
        if (maxTurn and max(scores) == 1) or (not maxTurn and min(scores) == -1):
            break
    return max(scores) if maxTurn else min(scores)



def make_move(current_board: board.Board, pl: board.Player):
        bestScore = -math.inf
        bestMove = None
        for move in current_board.possible_moves(pl):
            new_board = current_board.copy()
            new_board.move(pl, move)
            score = minimax(True, pl, new_board)
            if (score > bestScore):
                bestScore = score
                bestMove = move
        return bestMove


def random_move(current_board: board.Board, pl: board.Player) -> Optional[int]:
    moves = current_board.possible_moves(pl)
    return np.random.choice(moves, 1)[0]

def play():
    b = board.Board()
    while b.get_state() == board.BoardState.ONGOING:
        for pl in [board.Player.BLACK, board.Player.WHITE]:
                move = make_move(b, pl)
                b.move(pl, move)
                if b.get_state() != board.BoardState.ONGOING:
                    break
                print(b.to_np())