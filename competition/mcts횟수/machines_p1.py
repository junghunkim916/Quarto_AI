import numpy as np
import copy
import random
import math
from itertools import product

def check_win(board, pieces):
    # rows, cols, diagonals, 2x2 subgrid 체크
    def check_line(line):
        if 0 in line: return False
        chars = np.array([pieces[idx-1] for idx in line])
        for i in range(4):
            if len(set(chars[:,i])) == 1: return True
        return False
    for col in range(4):
        if check_line([board[row][col] for row in range(4)]): return True
    for row in range(4):
        if check_line([board[row][col] for col in range(4)]): return True
    if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3-i] for i in range(4)]): return True
    # 2x2 subgrid
    for r in range(3):
        for c in range(3):
            sub = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in sub:
                chars = [pieces[idx-1] for idx in sub]
                for i in range(4):
                    if len(set(char[i] for char in chars)) == 1: return True
    return False

class MCTSNode:
    def __init__(self, board, available_pieces, selected_piece, turn, pieces, parent=None):
        self.board = board
        self.available_pieces = available_pieces
        self.selected_piece = selected_piece
        self.turn = turn # 1 or 2 (player)
        self.pieces = pieces
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        actions = []
        # 놓을 위치 x,y
        if self.selected_piece is not None:
            locs = [(r,c) for r,c in product(range(4), range(4)) if self.board[r][c]==0]
            for loc in locs:
                actions.append(loc)
        return actions

    def is_terminal(self):
        return check_win(self.board, self.pieces) or all(self.board[r][c]!=0 for r in range(4) for c in range(4))

    def expand(self):
        action = self.untried_actions.pop()
        next_board = copy.deepcopy(self.board)
        next_board[action[0]][action[1]] = self.pieces.index(self.selected_piece)+1
        next_available = [p for p in self.available_pieces if p != self.selected_piece]
        if next_available:
            # 상대에게 줄 조각은 랜덤 (rollout의 확장성 때문에)
            next_piece = random.choice(next_available)
        else:
            next_piece = None
        child = MCTSNode(next_board, next_available, next_piece, 3-self.turn, self.pieces, parent=self)
        self.children.append((action, child))
        return child

    def best_child(self, c_param=1.4):
        choices = [
            (child, (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits))
            for (action, child) in self.children if child.visits > 0
        ]
        return max(choices, key=lambda x: x[1])[0]

    def rollout(self):
        board = copy.deepcopy(self.board)
        available_pieces = self.available_pieces[:]
        selected_piece = self.selected_piece
        turn = self.turn
        pieces = self.pieces

        while True:
            if check_win(board, pieces): return 3-turn # 전 턴의 승리
            locs = [(r,c) for r,c in product(range(4), range(4)) if board[r][c]==0]
            if not locs: return 0 # 무승부
            pos = random.choice(locs)
            board[pos[0]][pos[1]] = pieces.index(selected_piece)+1
            available_pieces = [p for p in available_pieces if p != selected_piece]
            if not available_pieces: return 0
            selected_piece = random.choice(available_pieces)
            turn = 3-turn

    def backpropagate(self, result, my_turn):
        self.visits += 1
        if result == my_turn:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result, my_turn)

def mcts_search(board, available_pieces, selected_piece, my_turn, pieces, n_sim):
    available_locs = [(r, c) for r, c in product(range(4), range(4)) if board[r][c] == 0]
    # 1. 즉시 승리
    for loc in available_locs:
        tmp_board = copy.deepcopy(board)
        tmp_board[loc[0]][loc[1]] = pieces.index(selected_piece) + 1
        if check_win(tmp_board, pieces):
            return loc

    # 2. 즉시 패배 방어 (상대가 다음 턴에 selected_piece로 승리할 수 있는 자리를 막음)
    # 상대가 받을 수 있는 모든 조각을 돌면서, 상대가 즉시 승리하는 자리가 있으면 그 자리 막기
    for loc in available_locs:
        tmp_board = copy.deepcopy(board)
        tmp_board[loc[0]][loc[1]] = pieces.index(selected_piece) + 1
        # 내 수를 놓고 난 다음, 상대에게 남은 조각 중 어떤 걸 줘도 상대가 바로 이길 수 있는지 체크
        opponent_pieces = [p for p in available_pieces if p != selected_piece]
        for op_piece in opponent_pieces:
            op_locs = [(r, c) for r, c in product(range(4), range(4)) if tmp_board[r][c] == 0]
            for op_loc in op_locs:
                tmp2_board = copy.deepcopy(tmp_board)
                tmp2_board[op_loc[0]][op_loc[1]] = pieces.index(op_piece) + 1
                if check_win(tmp2_board, pieces):
                    # 상대가 이길 수 있음 → 이 자리는 막아야 함
                    break
            else:
                continue
            break
        else:
            # 어떤 조각을 줘도 상대가 바로 못 이기는 자리 → 방어 최적 자리
            return loc

    # 3. 한 수 앞 강제승리: 내가 이 자리에 두면, 남은 어떤 조각을 줘도 내 다음 턴에 반드시 이기는 상황(선택적으로 추가)
    for loc in available_locs:
        tmp_board = copy.deepcopy(board)
        tmp_board[loc[0]][loc[1]] = pieces.index(selected_piece) + 1
        opponent_pieces = [p for p in available_pieces if p != selected_piece]
        all_forced_win = True
        for op_piece in opponent_pieces:
            op_locs = [(r, c) for r, c in product(range(4), range(4)) if tmp_board[r][c] == 0]
            can_block = False
            for op_loc in op_locs:
                tmp2_board = copy.deepcopy(tmp_board)
                tmp2_board[op_loc[0]][op_loc[1]] = pieces.index(op_piece) + 1
                # 내가 바로 못 이겨야 진짜 "forced win"이 아님
                next_locs = [(r, c) for r, c in product(range(4), range(4)) if tmp2_board[r][c] == 0]
                for next_loc in next_locs:
                    tmp3_board = copy.deepcopy(tmp2_board)
                    # 다음 턴에 내가 op_piece로 바로 못 이기는 경우가 하나라도 있으면 blocked
                    tmp3_board[next_loc[0]][next_loc[1]] = pieces.index(op_piece) + 1
                    if check_win(tmp3_board, pieces):
                        continue
                    else:
                        can_block = True
                        break
                if can_block:
                    break
            if can_block:
                all_forced_win = False
                break
        if all_forced_win:
            return loc

    # 4. 그 외에는 MCTS로 진행
    root = MCTSNode(copy.deepcopy(board), available_pieces[:], selected_piece, my_turn, pieces)
    for _ in range(n_sim):
        node = root
        while node.untried_actions == [] and node.children:
            node = node.best_child()
        if node.untried_actions:
            node = node.expand()
        result = node.rollout()
        node.backpropagate(result, my_turn)
    visits = [(action, child.visits) for (action, child) in root.children]
    best_action = max(visits, key=lambda x: x[1])[0] if visits else None
    return best_action

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        # 상대에게 줄 조각 (여긴 랜덤)
        return random.choice(self.available_pieces)

    def place_piece(self, selected_piece):
        # MCTS로 최적의 위치를 찾음
        action = mcts_search(self.board, self.available_pieces, selected_piece, 1, self.pieces, n_sim=10000)
        if action is not None:
            return action
        # fallback: 랜덤
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        return random.choice(available_locs)