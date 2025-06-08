# machines_p1.py
import numpy as np
import random
import time
import math
import copy

# 게임에서 쓰는 16개의 piece 정의 (main.py 과 동일)
pieces = [(i, j, k, l)
          for i in range(2)
          for j in range(2)
          for k in range(2)
          for l in range(2)]

def get_place_actions(board):
    """빈 칸 좌표 리스트 반환"""
    return [(r, c) for r in range(board.shape[0])for c in range(board.shape[1]) if board[r][c] == 0]

def line_win(line):
    """한 줄(line)에 놓인 4개의 piece가 특정 속성 하나를 공유하는지"""
    if 0 in line:
        return False
    chars = [pieces[idx - 1] for idx in line]
    for i in range(4):
        if len({ch[i] for ch in chars}) == 1:
            return True
    return False

def check_win(board):
    """보드 전체에서 승리 조건(행,열,대각선,2×2서브그리드)을 검사"""
    # 행/열
    for r in range(4):
        if line_win([board[r][c] for c in range(4)]):
            return True
    for c in range(4):
        if line_win([board[r][c] for r in range(4)]):
            return True
    # 대각선
    if line_win([board[i][i] for i in range(4)]): return True
    if line_win([board[i][3 - i] for i in range(4)]): return True
    # 2×2 서브그리드
    for r in range(3):
        for c in range(3):
            sub = [board[r][c], board[r][c+1],
                   board[r+1][c], board[r+1][c+1]]
            if line_win(sub):
                return True
    return False

class Node:
    """MCTS 트리의 단일 노드"""
    def __init__(self, board, available_pieces, current_piece=None,
                 parent=None, current_player=1):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.current_piece = current_piece        # None이면 '선택' 단계, 아니면 '배치' 단계
        self.parent = parent
        self.current_player = current_player      # 이 노드에서 행동할 플레이어 (1 또는 2)
        self.children = []
        self.visits = 0
        self.wins = 0
        # 아직 시도하지 않은 액션을 모아둠
        if self.current_piece is None:
            # 선택 단계: available_pieces 중 하나 고르기
            self.untried_actions = self.available_pieces.copy()
        else:
            # 배치 단계: 빈 칸 좌표 중 하나 고르기
            self.untried_actions = get_place_actions(self.board)
        self.action = None  # 이 노드를 만든 액션 (piece tuple 또는 (r,c) 좌표)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        """확장(Expansion): untried_actions 중 하나를 꺼내서 자식 노드 생성"""
        action = self.untried_actions.pop()
        if self.current_piece is None:
            # ---- 선택 단계: piece 고르기 ----
            new_avail = self.available_pieces.copy()
            new_avail.remove(action)
            child = Node(self.board,
                         new_avail,
                         current_piece=action,
                         parent=self,
                         current_player=self.current_player)
        else:
            # ---- 배치 단계: 좌표(action)에 current_piece 배치 ----
            new_board = self.board.copy()
            r, c = action
            new_board[r][c] = pieces.index(self.current_piece) + 1
            new_avail = self.available_pieces.copy()
            child = Node(new_board,
                         new_avail,
                         current_piece=None,
                         parent=self,
                         current_player=3 - self.current_player)
        child.action = action
        self.children.append(child)
        return child

    # def select_child(self, c_param=1.4):
    def select_child(self, round_cnt, initial_c = 1.4, decay_rate = 0.05):
        """선택(Selection): UCT 기준으로 자식 노드 중 하나 선택"""
        c_param = initial_c / (1 + decay_rate * round_cnt)
        best = None
        best_uct = -float('inf')
        for child in self.children:
            #C값 튜닝
            uct = (child.wins / child.visits) + \
                  c_param * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_uct:
                best_uct, best = uct, child
        return best

    def simulate(self):
        """시뮬레이션(Simulation): 랜덤 롤아웃해서 승자를 돌려줌"""
        sim_board = self.board.copy()
        sim_avail = self.available_pieces.copy()
        sim_piece = self.current_piece
        player = self.current_player
        while True:
            # 1) 선택 단계: piece 고르기
            if sim_piece is None and not sim_avail:
                return None
            if sim_piece is None:
                sim_piece = random.choice(sim_avail)
                sim_avail.remove(sim_piece)
            # 2) 배치 단계: 무작위 좌표에 놓기
            moves = get_place_actions(sim_board)
            if not moves:
                return None
            r, c = random.choice(moves)
            sim_board[r][c] = pieces.index(sim_piece) + 1
            # 승리 검사
            if check_win(sim_board):
                return player
            # 무승부 검사
            if not sim_avail:
                return None
            # 턴 교대
            player = 3 - player
            sim_piece = None

    def backpropagate(self, result):
        """역전파(Backpropagation): 결과를 따라 루트까지 방문/승리 통계 업데이트"""
        self.visits += 1
        # result가 이 노드의 부모에서 행동했던 플레이어와 같으면 승리로 간주
        # (선택 단계에서는 current_player, 배치에서는 parent.current_player 기준)
        win_player = self.parent.current_player if self.parent else self.current_player
        if result == win_player:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    """4단계를 적용한 MCTS 메인 클래스"""
    def __init__(self, board, available_pieces, current_piece=None, time_budget=0.5):
        self.root_board = board.copy()
        self.root_available = available_pieces.copy()
        self.root_piece = current_piece
        self.time_budget = time_budget

    def search_actions(self, for_place=False):
        """for_place=False → 선택 단계 (piece), True → 배치 단계 ((r,c) 좌표)"""
        """for_place=False → piece 선택, True → 배치"""
        # 즉시 승/패 먼저 점검
        if for_place:
            # 배치 단계: 이기는 수가 있는지 확인
            for r, c in get_place_actions(self.root_board):
                temp_board = self.root_board.copy()
                temp_board[r][c] = pieces.index(self.root_piece) + 1
                if check_win(temp_board):
                    return (r, c)
        else:
            # 선택 단계: 상대가 이길 수 있는 piece 피하기
            safe_pieces = []
            for piece in self.root_available:
                temp_board = self.root_board.copy()
                for r, c in get_place_actions(temp_board):
                    temp_board2 = temp_board.copy()
                    temp_board2[r][c] = pieces.index(piece) + 1
                    if check_win(temp_board2):
                        break  # 이 piece로 상대가 바로 이김
                else:
                    safe_pieces.append(piece)
            if safe_pieces:
                # 안전한 piece 중에서만 MCTS 실행
                temp_available = safe_pieces
            else:
                # 어차피 다 지는 상황이면 원래대로
                temp_available = self.root_available
                
        #본 코드 시작
        end_time = time.time() + self.time_budget
        init_player = 2 if for_place else 1
        root = Node(self.root_board,
                    temp_available if not for_place else self.root_available,
                    # self.root_available,
                    current_piece=self.root_piece if for_place else None,
                    parent=None,
                    current_player=init_player)

        round_cnt = 0
        while time.time() < end_time:
            node = root
            # 1) Selection
            while node.is_fully_expanded() and node.children:
                #param 전달 시 C값 튜닝
                node = node.select_child(round_cnt)
            # 2) Expansion
            if not node.is_fully_expanded():
                node = node.expand()
            # 3) Simulation
            result = node.simulate()
            # 4) Backpropagation
            node.backpropagate(result)
            # 탐색 횟수 증가
            round_cnt += 1 

        # 방문 횟수(visits)가 가장 많은 액션을 반환
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

# P1 클래스: piece 선택과 배치를 각각 MCTS로 수행
class P1:
    def __init__(self, board, available_pieces):
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        # 상대가 놓을 piece를 고름 → for_place=False
        mcts = MCTS(self.board, self.available_pieces, time_budget=0.5)
        return mcts.search_actions(for_place=False)

    def place_piece(self, selected_piece):
        # 주어진 piece를 어디에 놓을지 고름 → for_place=True
        mcts = MCTS(self.board, self.available_pieces,
                    current_piece=selected_piece,
                    time_budget=1.0)
        return mcts.search_actions(for_place=True)