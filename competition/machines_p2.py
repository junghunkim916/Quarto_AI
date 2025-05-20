import numpy as np
import random
from itertools import product
import time
import math

class P2:
    """
    Quarto 플레이어: 기본 Min-Max 알고리즘 구현
    """
    def __init__(self, board, available_pieces):
        # board: 4×4 numpy array, 0=empty, 1~16=piece index
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board.copy()
        self.available_pieces = list(available_pieces)
        self.max_depth = 2  # Min-Max 탐색 깊이

    def select_piece(self):
        """
        상대에게 줄 말을 Min-Max로 선택
        Min 단계: 상대가 말을 놓아 얻을 수 있는 최대 점수를 최소화
        """
        # 첫 수: 무작위 선택
        if np.count_nonzero(self.board) == 0:
            return random.choice(self.available_pieces)
        # 빈칸 및 인접 후보군 계산
        empties = self._empty_squares(self.board)
        # 인접 후보군: 기존 돌과 인접한 빈 칸만
        neighbors = []
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for (r,c) in empties:
            for dr,dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < 4 and 0 <= nc < 4 and self.board[nr,nc] != 0:
                    neighbors.append((r,c))
                    break
        candidate_locs = neighbors if neighbors else empties

        best_pieces = []
        best_score = math.inf  # P1 입장에선 상대 최고 점수의 최소화
        # 각 piece 후보에 대해 - 일단 지금은 모든 이용가능한 말들 비교 => 후보를 좁혀야 할 것 같음. (MoveGenerator)
        for piece in self.available_pieces:
            # 상대가 이 말로 얻을 수 있는 최대 점수 계산
            opponent_best = -math.inf
            #일단 말을 모든 빈칸에 둔다. 둔 후 각각 평가할 것.
            for loc in candidate_locs:
                b2 = self.board.copy()
                idx = self.available_pieces.index(piece) + 1
                b2[loc] = idx
                score = self._evaluate(b2)  # 상대 입장에서의 평가 (Evaluate Function)
                #상대가 이 말로 얻을 수 있는 최고점 - 즉, 최적의 loc을 둿을때의 가치
                if score > opponent_best:
                    opponent_best = score
            # opponenet_best 중 최소인 piece 선택 - 상대가 택했을때 점수가 가장 낮을 말 선택
            if opponent_best < best_score:
                best_score = opponent_best
                best_pieces.append(piece)
            elif opponent_best == best_score:
                best_pieces.append(piece)
        #최종 선택
        chosen_piece = random.choice(best_pieces) if best_pieces else random.choice(self.available_pieces)

        board_score = self._evaluate(self.board)
        print(f"[DEBUG select_piece] selected piece={chosen_piece}, best_score={best_score}")
        print(f"[DEBUG select_piece] board before selection:\n{self.board}")
        print(f"[DEBUG select_piece] board heuristic score: {board_score}")
        time.sleep(0.5)
        return chosen_piece


    def place_piece(self, selected_piece):
        """
        받은 말을 Min-Max 없이 가장 높은 heuristic 위치에 놓기
        """
        # available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        empties = self._empty_squares(self.board)
        neighbors = []
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for (r,c) in empties:
            for dr,dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < 4 and 0 <= nc < 4 and self.board[nr,nc] != 0:
                    neighbors.append((r,c))
                    break
        if neighbors:
            empties = neighbors
        # 첫 수 중앙 2*2그리드 우선 배치
        if not self.board.any():
            centers = [(1,1),(1,2),(2,1),(2,2)]
            for c in centers:
                if c in empties:
                    return c

        best_locs = []
        best_score = -np.inf
        #모든 가능한 칸마다 선택된 말을 놓아보며 최대가치를 계산함.
        for loc in empties:
            b2 = self.board.copy()
            idx = self.available_pieces.index(selected_piece) + 1
            b2[loc] = idx
            #이때 보드의 가치를 평가 -> 내가 이기면 +, 지면 -
            score = self._evaluate(b2)
            if score > best_score:
                best_score = score
                best_locs = [loc]
            elif score == best_score:
                best_locs.append(loc)
        #최종결정 
        chosen = random.choice(best_locs) if best_locs else random.choice(empties)
        b_final = self.board.copy()
        idx_final = self.available_pieces.index(selected_piece) + 1
        b_final[chosen] = idx_final
        final_score = self._evaluate(b_final)
        print(f"[DEBUG place_piece] chosen loc={chosen}, best_score={best_score}")
        print(f"[DEBUG place_piece] board after placement:\
{b_final}")
        print(f"[DEBUG place_piece] board heuristic score: {final_score}")
        time.sleep(1)
        return chosen
    
    def _empty_squares(self, board):
        """빈 칸 좌표 목록"""
        return [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]


    def _evaluate(self, board):
        """
        휴리스틱 평가: 완성된 Quarto 라인 개수
        """
        score = 0
        lines = []
        # 행/열
        for i in range(4):
            lines.append(board[i, :])
            lines.append(board[:, i])
        # 대각선
        lines.append(board.diagonal())
        lines.append(np.fliplr(board).diagonal())
        for line in lines:
            if 0 not in line:
                attrs = np.array([tuple(map(int, format(x-1, '04b'))) for x in line])
                if np.any(np.all(attrs == attrs[0], axis=0)):
                    score += 1
        return score

