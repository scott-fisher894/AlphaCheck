
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
                if move in self.board.legal_moves:
                    moving_piece = self.board.piece_at(self.selected_square)  # Get the piece at the original square
                    if moving_piece and moving_piece.piece_type == chess.PAWN and (square // 8 in [0, 7]):
                        # Pawn promotion logic
                        promotion_choice = chess.QUEEN  # Temporary, replace with actual user choice
                        promotion_move = chess.Move(self.selected_square, square, promotion_choice)
                        self.board.push(promotion_move)
                    else:
                        self.board.push(move)
                    self.selected_square = None
                    self.legal_moves_for_selected_piece = []
                else:
                    self.selected_square = None  # Deselect if move is illegal
                    self.legal_moves_for_selected_piece = []


    def handle_pawn_promotion(self, move):
        self.waiting_for_promotion = True
        self.display_message("Pawn Promotion: Press 'q' for Queen, 'r' for Rook, 'b' for Bishop, 'n' for Knight")
        promotion_choice = self.wait_for_promotion_input()
        promotion_move = chess.Move(move.from_square, move.to_square, promotion_choice)
        self.board.push(promotion_move)

    def wait_for_promotion_input(self):
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return chess.QUEEN
                    elif event.key == pygame.K_r:
                        return chess.ROOK
                    elif event.key == pygame.K_b:
                        return chess.BISHOP
                    elif event.key == pygame.K_n:
                        return chess.KNIGHT
        
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