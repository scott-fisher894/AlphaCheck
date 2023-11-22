
import chess
import pygame

WIDTH = HEIGHT = 900
DIMENSION = 8  # Chess board is an 8x8 square
SQ_SIZE = HEIGHT // DIMENSION

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.images = {}
        self.load_images()

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
                    y = (i // 8) * SQ_SIZE
                    # Centering the piece in the square
                    piece_image = self.images[piece_image_key]
                    surface.blit(piece_image, (x + (SQ_SIZE - piece_image.get_width()) // 2,
                                            y + (SQ_SIZE - piece_image.get_height()) // 2))
                    