# import numpy as np
# import random
# from itertools import product

# import time

# class P1():
#     def __init__(self, board, available_pieces):
#         self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
#         self.board = board # Include piece indices. 0:empty / 1~16:piece
#         self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
#     #기존 코드 
    # def select_piece(self):
    #     # Make your own algorithm here

    #     time.sleep(0.5) # Check time consumption (Delete when you make your algorithm)

    #     return random.choice(self.available_pieces)

    # def place_piece(self, selected_piece):
    #     # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        
    #     # Available locations to place the piece
    #     available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]

    #     # Make your own algorithm here

    #     time.sleep(1) # Check time consumption (Delete when you make your algorithm)
        
    #     return random.choice(available_locs)
    
# import numpy as np
# import random
# from itertools import product

# import time

# class P1():
#     def __init__(self, board, available_pieces):
#         self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
#         self.board = board # Include piece indices. 0:empty / 1~16:piece
#         self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
#     def select_piece(self):
#         # Make your own algorithm here

#         time.sleep(0.5) # Check time consumption (Delete when you make your algorithm)

#         return random.choice(self.available_pieces)

#     def place_piece(self, selected_piece):
#         # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        
#         # Available locations to place the piece
#         available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]

#         # Make your own algorithm here

#         time.sleep(1) # Check time consumption (Delete when you make your algorithm)
        
#         return random.choice(available_locs)
import numpy as np
import random
from itertools import product
import time

class P1():
    def __init__(self, board, available_pieces):
        #4개의 binary특징을 가진 16개의 말 생성 
        self.board = board
        self.available_pieces = available_pieces
        #16개 말 생성 
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

    #인자로 받은 board에서 승리하는 줄이 있는지 확인 -> 있으면 그 줄에 점수 100점 부여 
    def evaluate_board(self, board):
        def check_line(line):
            #line을 입력받음 -> 빈 칸이 있는 경우 -> 아직 승리 불가능 
            if 0 in line:
                return 0
            #self.pieces[piece_idx - 1]는 실제 특성 4개 ((i,j,k,l) 형태의 튜플) 반환 => characteristics는 각 줄 4개의 말에 대한 4*4배열이 될것
            characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
            for i in range(4):
                #4*4배열에서 열 기준으로(특성 하나당 4개의 말 비교)슬라이스 후 집합연산 적용해 길이가 1(모두 같은 특성 보유) -> 100점 부여 (이긴 거임)
                if len(set(characteristics[:, i])) == 1:
                    return 100
            return 0

        score = 0
        for i in range(4):
            #보드의 4개 가로줄, 세로줄에 대해 각각 check_line수행
            score += check_line([board[i][j] for j in range(4)])
            score += check_line([board[j][i] for j in range(4)])
        #대각선도 체크하기 
        score += check_line([board[i][i] for i in range(4)])
        score += check_line([board[i][3 - i] for i in range(4)])
        return score
    #상대에게 줄 말 선택 
    def select_piece(self):
        start_time = time.time()
        min_opponent_score = float('inf')
        best_piece = random.choice(self.available_pieces)
        empty_locs = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

        for piece in self.available_pieces:
            worst_case = 0
            for r, c in empty_locs:
                simulated_board = self.board.copy()
                simulated_board[r][c] = self.pieces.index(piece) + 1
                score = self.evaluate_board(simulated_board)
                worst_case = max(worst_case, score)
            if worst_case < min_opponent_score:
                min_opponent_score = worst_case
                best_piece = piece

        end_time = time.time()
        print(f"[SELECT_PIECE] Time: {end_time - start_time:.3f}s | Chosen Piece: {best_piece} | Estimated Opponent Worst Case: {min_opponent_score}")
        return best_piece

    #상대에게 선택당한 말 배치 
    def place_piece(self, selected_piece):
        start_time = time.time()
        best_score = -float('inf')
        best_move = None
        empty_locs = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        selected_index = self.pieces.index(selected_piece) + 1

        for r, c in empty_locs:
            new_board = self.board.copy()
            new_board[r][c] = selected_index
            score = self.evaluate_board(new_board)
            if score > best_score:
                best_score = score
                best_move = (r, c)

        if not best_move:
            best_move = random.choice(empty_locs)

        end_time = time.time()
        print(f"[PLACE_PIECE] Time: {end_time - start_time:.3f}s | Placed at: {best_move} | Score: {best_score} | Piece: {selected_piece}")
        return best_move
