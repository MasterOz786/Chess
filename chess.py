import copy
import random

# Chess board representation
class ChessBoard:
    def __init__(self):
        # Initialize 8x8 board with starting position
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.white_to_move = True
        self.move_log = []
        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.in_check = False
        self.pins = []
        self.checks = []
        self.game_over = False
        self.winner = None

    def print_board(self):
        """Display the chessboard in the console."""
        print("\n   a b c d e f g h")
        print("  ----------------")
        for i in range(8):
            print(f"{8-i} |", end=" ")
            for j in range(8):
                print(self.board[i][j], end=" ")
            print(f"| {8-i}")
        print("  ----------------")
        print("   a b c d e f g h\n")

    def make_move(self, start, end):
        """Make a move from start to end position."""
        start_row, start_col = start
        end_row, end_col = end
        piece = self.board[start_row][start_col]
        captured = self.board[end_row][end_col]
        
        # Update board
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = '.'
        
        # Update king position if moved
        if piece.lower() == 'k':
            if self.white_to_move:
                self.white_king_pos = (end_row, end_col)
            else:
                self.black_king_pos = (end_row, end_col)
        
        # Log move
        self.move_log.append((start, end, piece, captured))
        self.white_to_move = not self.white_to_move
        
        # Check for game end
        self.check_game_state()

    def get_valid_moves(self):
        """Get all valid moves for the current player."""
        moves = []
        self.pins = []
        self.checks = []
        king_pos = self.white_king_pos if self.white_to_move else self.black_king_pos
        
        # Check for pins and checks
        self.check_pins_and_checks(king_pos)
        
        if self.in_check:
            if len(self.checks) == 1:
                moves = self.get_check_moves()
            else:
                # Double check, only king moves allowed
                moves = self.get_king_moves(king_pos)
        else:
            for row in range(8):
                for col in range(8):
                    piece = self.board[row][col]
                    if (self.white_to_move and piece.isupper()) or \
                       (not self.white_to_move and piece.islower()):
                        moves.extend(self.get_piece_moves((row, col), piece))
        
        return moves

    def check_pins_and_checks(self, king_pos):
        """Check for pins and checks affecting the king."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        king_row, king_col = king_pos
        self.in_check = False
        
        for direction in directions:
            d_row, d_col = direction
            possible_pin = None
            for i in range(1, 8):
                end_row = king_row + d_row * i
                end_col = king_col + d_col * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    piece = self.board[end_row][end_col]
                    if piece != '.':
                        if (self.white_to_move and piece.isupper()) or \
                           (not self.white_to_move and piece.islower()):
                            if possible_pin is None:
                                possible_pin = (end_row, end_col)
                            else:
                                break
                        else:
                            piece_type = piece.lower()
                            if (piece_type == 'r' and direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]) or \
                               (piece_type == 'b' and direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]) or \
                               (piece_type == 'q') or \
                               (i == 1 and piece_type == 'p' and \
                                ((self.white_to_move and direction in [(-1, -1), (-1, 1)]) or \
                                 (not self.white_to_move and direction in [(1, -1), (1, 1)]))):
                                if possible_pin:
                                    self.pins.append((possible_pin, (end_row, end_col)))
                                else:
                                    self.checks.append((end_row, end_col, d_row, d_col))
                                    self.in_check = True
                            break
                else:
                    break

    def get_piece_moves(self, pos, piece):
        """Get valid moves for a specific piece."""
        moves = []
        row, col = pos
        piece_type = piece.lower()
        
        # Check if piece is pinned
        for pin in self.pins:
            if pin[0] == (row, col):
                return self.get_pinned_moves(pin, pos, piece)
        
        if piece_type == 'p':
            moves = self.get_pawn_moves(row, col, self.white_to_move)
        elif piece_type == 'r':
            moves = self.get_rook_moves(row, col)
        elif piece_type == 'n':
            moves = self.get_knight_moves(row, col)
        elif piece_type == 'b':
            moves = self.get_bishop_moves(row, col)
        elif piece_type == 'q':
            moves = self.get_queen_moves(row, col)
        elif piece_type == 'k':
            moves = self.get_king_moves((row, col))
        
        return [(pos, end) for end in moves]

    def get_pinned_moves(self, pin, pos, piece):
        """Get valid moves for a pinned piece."""
        moves = []
        pin_row, pin_col = pin[0]
        attacker_row, attacker_col = pin[1]
        piece_type = piece.lower()
        
        # Calculate direction from pinned piece to attacker
        d_row = attacker_row - pin_row
        d_col = attacker_col - pin_col
        
        # Normalize direction
        if d_row != 0:
            d_row = d_row // abs(d_row) if d_row != 0 else 0
        if d_col != 0:
            d_col = d_col // abs(d_col) if d_col != 0 else 0
            
        # Get valid squares along pin direction (including attacker position)
        valid_squares = [(attacker_row, attacker_col)]
        curr_row, curr_col = pin_row - d_row, pin_col - d_col
        while 0 <= curr_row < 8 and 0 <= curr_col < 8:
            if (curr_row, curr_col) == (self.white_king_pos if self.white_to_move else self.black_king_pos):
                break
            valid_squares.append((curr_row, curr_col))
            curr_row -= d_row
            curr_col -= d_col
        
        # Get all possible moves for the piece without considering pins
        if piece_type == 'p':
            possible_moves = self.get_pawn_moves(pin_row, pin_col, self.white_to_move)
        elif piece_type == 'r':
            possible_moves = self.get_rook_moves(pin_row, pin_col)
        elif piece_type == 'n':
            possible_moves = self.get_knight_moves(pin_row, pin_col)
        elif piece_type == 'b':
            possible_moves = self.get_bishop_moves(pin_row, pin_col)
        elif piece_type == 'q':
            possible_moves = self.get_queen_moves(pin_row, pin_col)
        else:  # King shouldn't be pinned
            return []
        
        # Filter moves to only those along the pin direction
        for move in possible_moves:
            if move in valid_squares:
                moves.append((pos, move))
        
        return moves

    def get_pawn_moves(self, row, col, is_white):
        """Get valid pawn moves."""
        moves = []
        direction = -1 if is_white else 1
        start_row = 6 if is_white else 1
        
        # Move forward
        if 0 <= row + direction < 8 and self.board[row + direction][col] == '.':
            moves.append((row + direction, col))
            if row == start_row and self.board[row + 2 * direction][col] == '.':
                moves.append((row + 2 * direction, col))
        
        # Capture diagonally
        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_col < 8 and 0 <= row + direction < 8:
                target = self.board[row + direction][new_col]
                if target != '.' and \
                   ((is_white and target.islower()) or (not is_white and target.isupper())):
                    moves.append((row + direction, new_col))
        
        return moves

    def get_rook_moves(self, row, col):
        """Get valid rook moves."""
        return self.get_sliding_moves(row, col, [(-1, 0), (1, 0), (0, -1), (0, 1)])

    def get_bishop_moves(self, row, col):
        """Get valid bishop moves."""
        return self.get_sliding_moves(row, col, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def get_queen_moves(self, row, col):
        """Get valid queen moves."""
        return self.get_sliding_moves(row, col, [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])

    def get_sliding_moves(self, row, col, directions):
        """Get moves for sliding pieces (rook, bishop, queen)."""
        moves = []
        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + d_row * i, col + d_col * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.':
                        moves.append((new_row, new_col))
                    else:
                        if (self.white_to_move and target.islower()) or \
                           (not self.white_to_move and target.isupper()):
                            moves.append((new_row, new_col))
                        break
                else:
                    break
        return moves

    def get_knight_moves(self, row, col):
        """Get valid knight moves."""
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for d_row, d_col in knight_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == '.' or \
                   (self.white_to_move and target.islower()) or \
                   (not self.white_to_move and target.isupper()):
                    moves.append((new_row, new_col))
        return moves

    def get_king_moves(self, pos):
        """Get valid king moves."""
        moves = []
        row, col = pos
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for d_row, d_col in king_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == '.' or \
                   (self.white_to_move and target.islower()) or \
                   (not self.white_to_move and target.isupper()):
                    # Check if move puts king in check
                    temp_board = copy.deepcopy(self)
                    temp_board.board[new_row][new_col] = temp_board.board[row][col]
                    temp_board.board[row][col] = '.'
                    temp_board.white_to_move = not temp_board.white_to_move
                    if not temp_board.is_square_attacked((new_row, new_col)):
                        moves.append((new_row, new_col))
        return moves

    def get_check_moves(self):
        """Get valid moves when in check."""
        moves = []
        king_pos = self.white_king_pos if self.white_to_move else self.black_king_pos
        check_row, check_col, d_row, d_col = self.checks[0]  # Correct unpacking
        
        # Generate king moves as (king_pos, end) tuples
        king_moves = self.get_king_moves(king_pos)
        moves.extend([(king_pos, end) for end in king_moves])
        
        # Block or capture checking piece
        blocking_squares = [
            (check_row - d_row * i, check_col - d_col * i) 
            for i in range(1, 8) 
            if 0 <= check_row - d_row * i < 8 and 0 <= check_col - d_col * i < 8
        ]
        blocking_squares.append((check_row, check_col))
        
        # Collect valid moves from other pieces
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (self.white_to_move and piece.isupper()) or (not self.white_to_move and piece.islower()):
                    piece_moves = self.get_piece_moves((row, col), piece)
                    for move in piece_moves:
                        _, end_pos = move
                        if end_pos in blocking_squares:
                            moves.append(move)
        
        return moves

    def is_square_attacked(self, pos):
        """Check if a square is attacked by opponent pieces."""
        row, col = pos
        temp_white_to_move = self.white_to_move
        self.white_to_move = not self.white_to_move
        
        # Check knight attacks
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for d_row, d_col in knight_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = self.board[new_row][new_col]
                if piece.lower() == 'n' and \
                   ((self.white_to_move and piece.isupper()) or \
                    (not self.white_to_move and piece.islower())):
                    self.white_to_move = temp_white_to_move
                    return True
        
        # Check sliding pieces (rook, bishop, queen)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + d_row * i, col + d_col * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    piece = self.board[new_row][new_col]
                    if piece != '.':
                        piece_type = piece.lower()
                        if ((self.white_to_move and piece.isupper()) or \
                            (not self.white_to_move and piece.islower())):
                            if (piece_type == 'r' and d_row * d_col == 0) or \
                               (piece_type == 'b' and d_row * d_col != 0) or \
                               (piece_type == 'q'):
                                self.white_to_move = temp_white_to_move
                                return True
                            break
                else:
                    break
        
        # Check pawn attacks
        pawn_dirs = [(-1, -1), (-1, 1)] if self.white_to_move else [(1, -1), (1, 1)]
        for d_row, d_col in pawn_dirs:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = self.board[new_row][new_col]
                if piece.lower() == 'p' and \
                   ((self.white_to_move and piece.isupper()) or \
                    (not self.white_to_move and piece.islower())):
                    self.white_to_move = temp_white_to_move
                    return True
        
        # Check king attacks
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for d_row, d_col in king_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = self.board[new_row][new_col]
                if piece.lower() == 'k' and \
                   ((self.white_to_move and piece.isupper()) or \
                    (not self.white_to_move and piece.islower())):
                    self.white_to_move = temp_white_to_move
                    return True
        
        self.white_to_move = temp_white_to_move
        return False

    def check_game_state(self):
        """Check for checkmate or stalemate."""
        moves = self.get_valid_moves()
        if not moves:
            king_pos = self.white_king_pos if self.white_to_move else self.black_king_pos
            if self.is_square_attacked(king_pos):
                self.game_over = True
                self.winner = 'Black' if self.white_to_move else 'White'
            else:
                self.game_over = True
                self.winner = 'Draw'

    def evaluate_board(self):
        """Evaluate the board position."""
        material = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        score = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':
                    value = material[piece.lower()]
                    score += value if piece.isupper() else -value
        return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """Minimax algorithm with alpha-beta pruning."""
    if depth == 0 or board.game_over:
        return board.evaluate_board(), None
    
    moves = board.get_valid_moves()
    if not moves:
        return board.evaluate_board(), None
    
    best_move = random.choice(moves)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            temp_board = copy.deepcopy(board)
            temp_board.make_move(move[0], move[1])
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            temp_board = copy.deepcopy(board)
            temp_board.make_move(move[0], move[1])
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def pos_to_coord(pos):
    """Convert board position to chess coordinate."""
    row, col = pos
    return f"{chr(97 + col)}{8 - row}"

def coord_to_pos(coord):
    """Convert chess coordinate to board position."""
    col = ord(coord[0].lower()) - 97
    row = 8 - int(coord[1])
    return (row, col)

def main():
    """Main game loop."""
    board = ChessBoard()
    print("Welcome to Chess!")
    print("Enter moves in format: e2e4 (start square to end square)")
    print("Enter 'quit' to exit")
    
    while not board.game_over:
        board.print_board()
        if board.white_to_move:
            move = input("Your move: ")
            if move.lower() == 'quit':
                break
            try:
                start = coord_to_pos(move[:2])
                end = coord_to_pos(move[2:])
                if (start, end) in board.get_valid_moves():
                    board.make_move(start, end)
                else:
                    print("Invalid move! Try again.")
                    continue
            except:
                print("Invalid input format! Use e2e4 format.")
                continue
        else:
            print("Computer thinking...")
            _, best_move = minimax(board, 3, float('-inf'), float('inf'), False)
            if best_move:
                start, end = best_move
                print(f"Computer moves: {pos_to_coord(start)}{pos_to_coord(end)}")
                board.make_move(start, end)
    
    board.print_board()
    if board.winner:
        print(f"Game Over! {board.winner} wins!" if board.winner != 'Draw' else "Game Over! It's a draw!")

if __name__ == "__main__":
    main()