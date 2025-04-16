import tkinter as tk
from tkinter import messagebox
import os
from PIL import Image, ImageTk
import copy
import random
import threading
from queue import Queue, Empty

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
            d_row = d_row // abs(d_row)
        if d_col != 0:
            d_col = d_col // abs(d_col)
            
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
        check_row, check_col, d_row, d_col = self.checks[0]
        
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
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (self.white_to_move and piece.isupper()) or \
                   (not self.white_to_move and piece.islower()):
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

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        self.board = ChessBoard()
        self.selected_square = None
        self.piece_images = {}
        self.square_size = 64
        self.move_queue = Queue()
        self.setup_gui()
        self.load_pieces()
        self.update_board()

    def setup_gui(self):
        """Set up the GUI components."""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # Chess board frame
        self.board_frame = tk.Frame(self.main_frame)
        self.board_frame.pack(side=tk.LEFT)

        # Create squares
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        for row in range(8):
            for col in range(8):
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                square_frame = tk.Frame(self.board_frame, 
                                      width=self.square_size, 
                                      height=self.square_size,
                                      bg=color,
                                      borderwidth=1,
                                      relief='solid')
                square_frame.grid(row=row, column=col)
                square_frame.grid_propagate(False)
                
                square = tk.Label(square_frame, bg=color)
                square.place(relx=0.5, rely=0.5, anchor='center')
                square_frame.bind('<Button-1>', lambda e, r=row, c=col: self.square_clicked(r, c))
                
                self.squares[row][col] = square
                self.squares[row][col].frame = square_frame

        # Rank and file labels
        for col in range(8):
            label = tk.Label(self.board_frame, text=chr(97 + col), font=('Arial', 10))
            label.grid(row=8, column=col, sticky='n')
        
        for row in range(8):
            label = tk.Label(self.board_frame, text=str(8 - row), font=('Arial', 10))
            label.grid(row=row, column=8, sticky='w')

        # Info frame
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(side=tk.LEFT, padx=20)

        # Turn and status labels
        self.turn_label = tk.Label(self.info_frame, text="White's turn", font=('Arial', 14))
        self.turn_label.pack(pady=10)

        self.status_label = tk.Label(self.info_frame, text="", font=('Arial', 12))
        self.status_label.pack(pady=5)

        # Move history
        self.history_frame = tk.Frame(self.info_frame)
        self.history_frame.pack(pady=10)
        
        tk.Label(self.history_frame, text="Move History:", font=('Arial', 12)).pack()
        self.history_text = tk.Text(self.history_frame, width=20, height=15)
        self.history_text.pack()

    def load_pieces(self):
        """Load chess piece images."""
        piece_chars = {'p': 'Pawn', 'r': 'Rook', 'n': 'Knight', 
                      'b': 'Bishop', 'q': 'Queen', 'k': 'King'}
        
        if not os.path.exists('pieces'):
            os.makedirs('pieces')
            print("Created 'pieces' directory. Add chess piece images for better UI.")
            
        piece_size = int(self.square_size * 0.8)

        for piece_char, piece_name in piece_chars.items():
            try:
                image = Image.open(f'pieces/w{piece_name}.png')
                image = image.resize((piece_size, piece_size))
                self.piece_images[piece_char.upper()] = ImageTk.PhotoImage(image)
            except:
                self.piece_images[piece_char.upper()] = piece_char.upper()

            try:
                image = Image.open(f'pieces/b{piece_name}.png')
                image = image.resize((piece_size, piece_size))
                self.piece_images[piece_char] = ImageTk.PhotoImage(image)
            except:
                self.piece_images[piece_char] = piece_char

    def update_board(self, changed_squares=None):
        """Update the GUI board with current game state."""
        if changed_squares is None:
            changed_squares = [(r, c) for r in range(8) for c in range(8)]
        
        for row, col in changed_squares:
            piece = self.board.board[row][col]
            square = self.squares[row][col]
            
            square.configure(image='', text='')
            
            if piece != '.' and piece in self.piece_images:
                if isinstance(self.piece_images[piece], str):
                    square.configure(text=self.piece_images[piece], font=('Arial', 24))
                else:
                    square.configure(image=self.piece_images[piece])
                    square.image = self.piece_images[piece]
        
        self.turn_label.config(text="White's turn" if self.board.white_to_move else "Black's turn")
        self.root.update_idletasks()

    def square_clicked(self, row, col):
        """Handle square click events."""
        if not self.board.white_to_move or self.board.game_over:
            return

        if self.selected_square is None:
            piece = self.board.board[row][col]
            if piece.isupper():
                self.selected_square = (row, col)
                self.highlight_square(row, col)
                self.highlight_valid_moves(row, col)
        else:
            start_row, start_col = self.selected_square
            move = ((start_row, start_col), (row, col))
            
            valid_moves = self.board.get_valid_moves()
            if move in valid_moves:
                self.make_move(move)
                self.clear_highlights()
                self.make_computer_move()
            
            self.selected_square = None
            self.clear_highlights()

    def highlight_square(self, row, col):
        """Highlight selected square."""
        self.squares[row][col].frame.config(bg='#646E40')
        self.squares[row][col].config(bg='#646E40')

    def highlight_valid_moves(self, row, col):
        """Highlight valid moves for selected piece."""
        valid_moves = self.board.get_valid_moves()
        for start, end in valid_moves:
            if start == (row, col):
                end_row, end_col = end
                self.squares[end_row][end_col].frame.config(bg='#779952')
                self.squares[end_row][end_col].config(bg='#779952')

    def clear_highlights(self):
        """Clear all highlights from the board."""
        for row in range(8):
            for col in range(8):
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.squares[row][col].frame.config(bg=color)
                self.squares[row][col].config(bg=color)

    def make_move(self, move):
        """Make a move and update the game state."""
        start, end = move
        self.board.make_move(start, end)
        self.update_board([start, end])
        self.add_move_to_history(start, end)
        self.check_game_end()

    def compute_best_move(self):
        """Compute AI's move in a separate thread."""
        _, best_move = minimax(self.board, 3, float('-inf'), float('inf'), False)
        self.move_queue.put(best_move)

    def check_move_queue(self):
        """Check if AI move is ready."""
        try:
            best_move = self.move_queue.get_nowait()
            self.status_label.config(text="")
            if best_move:
                self.make_move(best_move)
        except Empty:
            self.root.after(100, self.check_move_queue)

    def make_computer_move(self):
        """Start AI move computation in a separate thread."""
        if not self.board.game_over:
            self.status_label.config(text="AI is thinking...")
            threading.Thread(target=self.compute_best_move, daemon=True).start()
            self.root.after(100, self.check_move_queue)

    def add_move_to_history(self, start, end):
        """Add move to history display."""
        piece = self.board.move_log[-1][2]
        captured = self.board.move_log[-1][3]
        
        move_text = f"{piece} {pos_to_coord(start)}"
        if captured != '.':
            move_text += f"x{captured} {pos_to_coord(end)}"
        else:
            move_text += f"-{pos_to_coord(end)}"
            
        if len(self.board.move_log) % 2 == 1:
            move_number = (len(self.board.move_log) + 1) // 2
            move_text = f"{move_number}. {move_text}"
        
        self.history_text.insert(tk.END, move_text + "\n")
        self.history_text.see(tk.END)

    def check_game_end(self):
        """Check if the game has ended."""
        if self.board.game_over:
            if self.board.winner == 'Draw':
                messagebox.showinfo("Game Over", "It's a draw!")
            else:
                messagebox.showinfo("Game Over", f"{self.board.winner} wins!")

def main():
    root = tk.Tk()
    chess_gui = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()