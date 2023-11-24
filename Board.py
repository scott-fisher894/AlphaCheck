
import chess
import pygame

WIDTH = HEIGHT = 900
DIMENSION = 8  # Chess board is an 8x8 square
SQ_SIZE = HEIGHT // DIMENSION

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.images = {}
        self.selected_square = None
        self.load_images()
        self.legal_moves_for_selected_piece = []
        self.highlight_color = (204, 204, 0)
        self.waiting_for_promotion = False

    def load_images(self):
        # Load images for each piece
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        for piece in pieces:
            for color in ['b', 'w']:
                filename = f"images/{color}{piece}.png"
                image = pygame.image.load(filename).convert_alpha()
                # Scale the image to fit the square size
                if piece == 'p':
                    scaled_image = pygame.transform.scale(image, (int(SQ_SIZE * .65), int(SQ_SIZE * .65)))
                else:
                    scaled_image = pygame.transform.scale(image, (int(SQ_SIZE * .75), int(SQ_SIZE * .75)))

                self.images[color + piece] = scaled_image

    def show_board(self, surface):
        colors = [(234, 235, 200), (119, 154, 88)]  # Light and dark square colors
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                pygame.draw.rect(surface, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Add code to draw pieces based on the python-chess board state
        for i in range(64):
            square = chess.SQUARES[i]
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol().lower()
                piece_image_key = ('w' if piece.color == chess.WHITE else 'b') + symbol
                if piece_image_key in self.images:
                    x = (i % 8) * SQ_SIZE
                    y = (7 - (i // 8)) * SQ_SIZE
                    # Centering the piece in the square
                    piece_image = self.images[piece_image_key]
                    surface.blit(piece_image, (x + (SQ_SIZE - piece_image.get_width()) // 2,
                                            y + (SQ_SIZE - piece_image.get_height()) // 2))
                
                # Highlight selected square
                if self.selected_square is not None:
                    x = (self.selected_square % 8) * SQ_SIZE
                    y = (7 - (self.selected_square // 8)) * SQ_SIZE
                    pygame.draw.rect(surface, self.highlight_color, pygame.Rect(x, y, SQ_SIZE, SQ_SIZE), 5)

                # Highlight legal moves
                for move in self.legal_moves_for_selected_piece:
                    target_square = move.to_square
                    x = (target_square % 8) * SQ_SIZE
                    y = (7 - (target_square // 8)) * SQ_SIZE
                    pygame.draw.rect(surface, self.highlight_color, pygame.Rect(x, y, SQ_SIZE, SQ_SIZE), 5)

    def handle_mouse_event(self, event):
        if event.button == 1:  # Left mouse button
            pos = pygame.mouse.get_pos()
            col = pos[0] // SQ_SIZE
            row = DIMENSION - 1 - (pos[1] // SQ_SIZE)
            square = chess.square(col, row)

            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.legal_moves_for_selected_piece = [move for move in self.board.legal_moves if move.from_square == self.selected_square]
            else:
                move = chess.Move(self.selected_square, square)
                # Check for potential promotion
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and square // 8 in [0, 7]:
                    self.waiting_for_promotion = True
                    self.promotion_start_square = self.selected_square  # Starting square of the pawn
                    self.promotion_target_square = square  # Target square for promotion
                    # No move is made yet, just waiting for promotion piece selection
                    # Create a promotion move with a temporary promotion choice
                    #temp_promotion_move = chess.Move(self.selected_square, square, chess.QUEEN)
                    #if temp_promotion_move in self.board.legal_moves:
                        # Handle actual promotion here (allow the player to choose the piece)
                    #    promotion_choice = chess.QUEEN  # Implement this method
                    #    promotion_move = chess.Move(self.selected_square, square, promotion_choice)
                    #    self.board.push(promotion_move)
                elif move in self.board.legal_moves:
                    self.board.push(move)
                self.selected_square = None
                self.legal_moves_for_selected_piece = []

    def handle_promotion_input(self, event):
        if event.key == pygame.K_q:
            self.execute_promotion(chess.QUEEN)
        elif event.key == pygame.K_r:
              self.execute_promotion(chess.ROOK)
        elif event.key == pygame.K_b:
               self.execute_promotion(chess.BISHOP)
        elif event.key == pygame.K_n:
               self.execute_promotion(chess.KNIGHT)
        #self.waiting_for_promotion = False

    def execute_promotion(self, promotion_piece):
        promotion_move = chess.Move(self.promotion_start_square, self.promotion_target_square, promotion_piece)
        self.board.push(promotion_move)
        self.waiting_for_promotion = False
        self.selected_square = None  # Reset the selected square
        self.promotion_square = None  # Reset the promotion square
  
    def get_status_text(self):
        if self.waiting_for_promotion:
            return "Pawn Promotion: 'q' Queen, 'n' Knight, 'r' Rook, 'b' Bishop"
        elif self.board.is_checkmate():
            return "Checkmate!"
        elif self.board.is_check():
            return "Check!"
        else:
            turn_text = "White's Turn" if self.board.turn == chess.WHITE else "Black's Turn"
            return turn_text