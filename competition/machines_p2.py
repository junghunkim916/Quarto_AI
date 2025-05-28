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
        self.max_depth = 3  # Min-Max 탐색 깊이
        
        # 휴리스틱 가중치 초기화
        self.w_diversity = 1.0
        self.w_threat    = 1.0
        self.w_potential = 1.0

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
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        ##
        nb = []
        #empty의 상하좌우 대각선에 말이 있는 경우 이웃에 추가. 이웃 배열이 후보군이 될 것.
        for r,c in empties:
            if any(0 <= r+dr < 4 and 0 <= c+dc < 4 and self.board[r+dr,c+dc]
                   for dr,dc in directions):
                nb.append((r,c))
        # candidate_locs = nb if nb else empties -> 작동을 잘 안하는듯?? 
        candidate_locs = empties
        
        ## location 후보는 결정되었고, 각 loc에 남은 말들을 놓았을 때 상대가 최대로 이익을 얻는 경우를 최소화 시키면 됨

        best_pieces = []
        best_score = math.inf  
        for piece in self.available_pieces:
            #P1의 최고점수 
            opponent_best = -math.inf
            immediate_loss = False
            #locaion 후보군 별로 보드를 카피해서 각각 min-max Tree를 생성
            for loc in candidate_locs:
                b2 = self.board.copy()
                idx = self.available_pieces.index(piece) + 1
                b2[loc] = idx
                #이렇게 다음 수를 놓았을 때 승리하는 경우 flag값 True로 바꾸기 
                if self._is_quarto(b2):
                    immediate_loss = True
                    break
                
                ## min-max로 심층 평가
                avail_next = self.available_pieces.copy()
                avail_next.remove(piece)

                # opponent(최대화) 차례로 시작: 깊이-1
                sc = self._minmax(b2, avail_next, self.max_depth-1, True, piece, -math.inf, math.inf)
                # min-max의 결과값이랑 P1의 기존 최댓값과 비교해 최댓값을 갱신
                opponent_best = max(opponent_best, sc)

            #즉시 패배 flag가 참이 되어 있으면 , 해당 조각은 제외하고 다음 조각을 본다
            if immediate_loss:
                continue 
            # 이 조각을 사용했을 때 상대가 얻을 수 있는 최댓값이 이전 조각을 사용했을 때의 최댓값보다 작으면 -> best_score를 갱신
            if opponent_best < best_score:
                best_score = opponent_best
                best_pieces = [piece]
            #같은 경우 최종 리턴 후보 리스트에 포함
            elif opponent_best == best_score:
                best_pieces.append(piece)
        #최종 선택 - 최종 리턴 후보 리스트에서 랜덤으로 뽑도록 한다. 후보에 없는 경우 그냥 랜덤으로 뽑도록 한다
        chosen_piece = random.choice(best_pieces) if best_pieces else random.choice(self.available_pieces)

        # #디버깅용- 휴리스틱을 이용한보드 가치 평가
        # board_score = self._evaluate(self.board)
        # print(f"[DEBUG select_piece] selected piece={chosen_piece}, best_score={best_score}")
        # print(f"[DEBUG select_piece] board before selection:\n{self.board}")
        # print(f"[DEBUG select_piece] board heuristic score: {board_score}")
        time.sleep(0.5)
        return chosen_piece


#내가 보드에 말을 둘 때, 상대방이 다음 말을 뒀을 때 얻는 이익이 최소화되도록 말을 둔다 -> 나의 이익이 최대화되도록 얻게 하기는 애매하다고 생각 -> 내가 다음 수를 뒀을 때 이기는 상황만 바로 말을 두도록 처리
    def place_piece(self, selected_piece):
        """
        받은 말을 Min-Max 없이 가장 높은 heuristic 위치에 놓기?
        """
        # 첫 수: 보드가 완전히 비어 있으면 중앙 2×2에서 랜덤 배치
        if not self.board.any():
            centers = [(1,1), (1,2), (2,1), (2,2)]
            valid = [c for c in centers if c in empties]
            return random.choice(valid)

        # available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        #비어있는 위치 추출
        empties = self._empty_squares(self.board)
        nb = []
        #둘 말은 상대가 정해줬으므로, 모든 location을 돌며 어디 놓는게 가치가 최대가 될지 판단하면 됨
        for loc in empties:
            b2 = self.board.copy()
            idx = self.available_pieces.index(selected_piece) + 1
            b2[loc] = idx
            #다음 수를 뒀을 때 내가 이기는지 
            if self._is_quarto(b2):
                print(f"[DEBUG place_piece] immediate win at {loc}")
                return loc
        
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        ###이웃된 location만 본다
        for r,c in empties:
            if any(0 <= r+dr < 4 and 0 <= c+dc < 4 and self.board[r+dr,c+dc]
                   for dr,dc in directions):
                nb.append((r,c))
        if nb:
            empties = nb
            
        best_locs = []
        #이 점수를 최대화하면 됨
        best_score = -np.inf
        #모든 가능한 칸마다 선택된 말을 놓아보며 최대가치를 계산함.
        for loc in empties:
            b2 = self.board.copy()
            idx = self.available_pieces.index(selected_piece) + 1
            b2[loc] = idx

            avail_next = self.available_pieces.copy()
            avail_next.remove(selected_piece)

            # opponent이 줄 다음 조각을 고르는 Min 단계
            # 각 반복마다 나온 말을 상대편이 둘 것 -> 두면서 worst score를 최소화한다
            worst_score = math.inf
            for opp in avail_next:
                tmp_avail = avail_next.copy()
                tmp_avail.remove(opp)
                sc = self._minmax(
                    b2, tmp_avail,
                    self.max_depth - 1,
                    True,
                    opp,
                    -math.inf,
                    math.inf
                )
                #이 location이 더욱 낮은 가치를 반환하는 경우 갱신 , 즉 나의 최대 이익보다 상대의 이익을 최소화하도록
                worst_score = min(worst_score, sc)


            if worst_score > best_score:
                best_score = worst_score
                best_locs = [loc]
            elif worst_score == best_score:
                best_locs.append(loc)
        #최종결정 
        chosen = random.choice(best_locs) if best_locs else random.choice(empties)
        
        #for debug
#         b_final = self.board.copy()
#         idx_final = self.available_pieces.index(selected_piece) + 1
#         b_final[chosen] = idx_final
#         final_score = self._evaluate(b_final)
#         print(f"[DEBUG place_piece] chosen loc={chosen}, best_score={best_score}")
#         print(f"[DEBUG place_piece] board after placement:\
# {b_final}")
#         print(f"[DEBUG place_piece] board heuristic score: {final_score}")
#         time.sleep(1)
        return chosen
    
    def _empty_squares(self, board):
        """빈 칸 좌표 목록"""
        return [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]

    #각 휴리스틱 함수들의 리턴값에 가중합을 구해 가치 리턴
    def _evaluate(self, board, piece=None, loc=None):
        # composite: diversity - threat + potential
        div = self._diversity(piece) if piece is not None else 0
        thr = self._threat(board)
        pot = self._potential(board, piece, loc) if (piece is not None and loc is not None) else 0
        return self.w_diversity * div - self.w_threat * thr + self.w_potential * pot
    
    #트리 전개 시 패배 회피, 승리 확정
    def _is_quarto(self, board):
        # 보드상에 퀘르토(완성) 상태가 있는지 확인
        # 1) 가로/세로/대각선/2*2grid 체크 - 라인 리스트 내의 각 라인 별로 검사
        for line in self._builtin_lines(board):
            # 라인에 0이 없는 경우 -> 다 채워진 경우
            if 0 not in line:
                # 해당 라인의 요소 x를 4비트 벡터로 바꾼 리스트를 attrs라 한다. [[E,S,T,P],...,[]]의 4*4 행렬 
                attrs = np.array([self._attr_vector(x) for x in line])
                if np.any(np.all(attrs == attrs[0], axis=0)):
                    return True

        return False
    ## 상하좌우 대각선의 라인리스트를 리턴
    def _builtin_lines(self, board):
        lines = []
        for i in range(4):
            lines.append(board[i, :])
            lines.append(board[:, i])
        lines.append(board.diagonal())
        lines.append(np.fliplr(board).diagonal())
        # 3) 2×2 정사각 블록
        for r in range(3):
            for c in range(3):
                square = [
                    board[r+0, c+0],
                    board[r+0, c+1],
                    board[r+1, c+0],
                    board[r+1, c+1],
                ]
                lines.append(square)
        return lines

    def _attr_vector(self, idx):
        return np.array(list(map(int, format(idx-1, '04b'))))

    def _diversity(self, piece):  # scoring #1
        #이미 말이 있는 위치 배열
        placed = [(r, c) for r, c in product(range(4), range(4)) if self.board[r, c] != 0]
        #말이 없는 경우 0반환
        if not placed:
            return 0
        #방금 놓은 piece
        vector_p = np.array(piece) # (E/I, S/N, T/F, J/P) 4비트 벡터
        total = 0
        #말이 있는 위치의 각 인덱스 r,c에 대해
        for r, c in placed:
            vector_q = self._attr_vector(self.board[r, c]) # 기존 보드 위 조각의 속성 벡터
            total += np.sum(vector_p != vector_q) #해밍 거리 계산해 총합 내기
        return total / len(placed) #4로 나눠 평균화해 리턴 -> 거리가 멀수록 좋은 것

    def _threat(self, board):  # scoring #2
        #위협도 측정 - 상대가 완성할 가능성 측정
        w3, w2 = 10, 3
        threat = 0
        #보드의 가로 세로 대각선 2*2grid에 대해 각 라인에 위치한 말들을 뽑아 nonzero라고 함
        for line in self._builtin_lines(board):
            nonzero = [x for x in line if x != 0]
            #k는 해당 라인에서 몇개나 뒀는지..
            k = len(nonzero)
            if 1 < k < 4:
                #같은 속성을 공유하는 말들이 3개 모인 경우 위협도 10 증가 , 2개 모인 경우 3증가
                attrs = np.array([self._attr_vector(x) for x in nonzero])
                if np.any(np.all(attrs == attrs[0], axis=0)):
                    threat += w3 if k == 3 else w2
        return threat

    def _potential(self, board, piece, loc):  # scoring #4
        #내가 이 수를 두었을 때, 승리에 얼마나 가까워졌는지 계산
        w3, w2 = 5, 1
        b2 = board.copy()
        idx = self.available_pieces.index(piece) + 1
        b2[loc] = idx
        pot = 0
        #수를 둔 다음, 라인을 평가해서 같은 속성 1개이상 갖는 말이 3개 모였으면 5점, 2개 모였으면 1점 추가
        for line in self._builtin_lines(b2):
            nonzero = [x for x in line if x != 0]
            k = len(nonzero)
            if k in (2, 3):
                # 2) 그 말들이 모두 같은 속성을 공유하는지 검사
                attrs = np.array([self._attr_vector(x) for x in nonzero])
                if np.any(np.all(attrs == attrs[0], axis=0)):
                    # 공통 속성이 있으면 잠재력 가중치 더하기
                    pot += (w3 if k == 3 else w2)
        return pot

    
    #보드, 가능한 말 리스트, 깊이, 누구턴인지, 이번에 두는 말, alpha-beta pruning용 변수
    def _minmax(self, board, available, depth, maximizing, current_piece, alpha=-math.inf, beta=math.inf):
        """
        알파-베타 프루닝 적용 Min-Max 재귀 함수
        - depth: 남은 깊이
        - maximizing: True면 우리(turn), False면 상대(turn)
        - current_piece: 이번에 둘 조각
        - alpha, beta: pruning 변수
        """
        #우리턴일때 이 말을 둬서 게임이 종료되면 최대 점수 리턴
        if self._is_quarto(board):
            return math.inf if maximizing else -math.inf

        # 종료 조건 - depths가 0이거나, 놓을 말이 없으면...-> 현재 보드의 가치를 평가해 리턴
        if depth == 0 or not available:
            return self._evaluate(board)
        #현재 보드에서 0인 위치를 모두 구한 리스트
        empties = self._empty_squares(board)

        #내 턴인 경우
        if maximizing:
            #각 호출마다 value초기화 -> 이걸 크게 만들어야 함 내 턴에는 
            value = -math.inf
            #모든 비어있는 location에 대해 반복
            for loc in empties:
                b2 = board.copy()
                idx = self.pieces.index(current_piece) + 1
                b2[loc] = idx

                # 즉시 승리 감지-돌을 한 개 놓은 뒤의 보드 검사 -> 더 들어가지 않도록
                # “놓자마자” 승리라면
                if self._is_quarto(b2):
                    return math.inf if maximizing else -math.inf
                # 다음 후보들 - 현재 놓은 말이 아닌 경우 다음에 놓을 수 있는 말들의 후보가 됨
                next_list = [p for p in available if p != current_piece]
                for nxt in next_list:
                    #각 loc에 대해 말들마다 돌아가며 minmax재귀호출
                    next_avail = available.copy()
                    next_avail.remove(nxt)
                    val = self._minmax(
                        b2, next_avail,
                        depth - 1,
                        # 내 턴인 경우 다음엔 상대 턴이므로
                        False,
                        nxt,
                        alpha, beta
                    )
                    # 현재 value의 최댓값과 더 깊이 내려가 얻은 가치를 비교해 최댓값 갱신
                    value = max(value, val)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        print(f"[DEBUG Prune-MAX] depth={depth} loc={loc} piece={current_piece} "
                              f"alpha={alpha:.2f} beta={beta:.2f} value={value:.2f}")
                        return value
            return value

        else:  # minimizing
            value = math.inf
            for loc in empties:
                b2 = board.copy()
                idx = self.pieces.index(current_piece) + 1
                b2[loc] = idx

                # 즉시 패배 감지
                if self._is_quarto(b2):
                    return -100

                next_list = [p for p in available if p != current_piece]
                for nxt in next_list:
                    next_avail = available.copy()
                    next_avail.remove(nxt)
                    val = self._minmax(
                        b2, next_avail,
                        depth - 1,
                        True,
                        nxt,
                        alpha, beta
                    )
                    value = min(value, val)
                    beta = min(beta, value)
                    if alpha >= beta:
                        print(f"[DEBUG Prune-MIN] depth={depth} loc={loc} piece={current_piece} "
                              f"alpha={alpha:.2f} beta={beta:.2f} value={value:.2f}")
                        return value
            return value
