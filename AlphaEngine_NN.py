import chess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChessEvaluationNN(nn.Module):
    def __init__(self):
        super(ChessEvaluationNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(-1, 32 * 8 * 8)  # Updated line
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(-1)
        return x


def convert_board_to_tensor(board):
    # Initialize an empty 8x8x4 representation
    board_tensor = np.zeros((8, 8, 4), dtype=np.float32)

    # Fill in the piece positions (first two channels)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for piece_type in piece_types:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == piece_type:
                channel = 0 if piece.color == chess.WHITE else 1
                board_tensor[square // 8, square % 8, channel] = piece_types.index(piece_type) + 1

    # Fill in the attack maps (last two channels)
    for move in board.legal_moves:
        if board.turn == chess.WHITE:
            board_tensor[move.to_square // 8, move.to_square % 8, 2] = 1
        else:
            board_tensor[move.to_square // 8, move.to_square % 8, 3] = 1

    # Convert numpy array to torch tensor
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
    board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
    board_tensor = board_tensor.permute(0, 3, 1, 2)  # Rearrange to [batch, channels, height, width]

    return board_tensor

# Load the trained model
model_path = 'chess_model.pth'  # Adjust the path if necessary
model = ChessEvaluationNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Evaluate Board using Neural Network
def evaluate(board, model):
    board_tensor = convert_board_to_tensor(board)
    with torch.no_grad():
        board_eval = model(board_tensor)
        return board_eval.item()

def alphabeta(board, depth, alpha, beta, maximizing_player, position_counts):
    if depth == 0 or board.is_game_over():
        position_counts[0] += 1
        return evaluate(board, model)

    if maximizing_player:
        max_eval = -9999
        for move in board.legal_moves:
            board.push(move)
            eval = alphabeta(board, depth - 1, alpha, beta, False, position_counts)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 9999
        for move in board.legal_moves:
            board.push(move)
            eval = alphabeta(board, depth - 1, alpha, beta, True, position_counts)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def select_best_move(board, depth):
    best_move = None
    best_value = -99999
    position_counts = [0]  # Position counter

    start_time = time.time()
    alpha = -10000
    beta = 10000
    for move in board.legal_moves:
        board.push(move)
        board_value = alphabeta(board, depth - 1, alpha, beta, False, position_counts)
        board.pop()
        if board_value > best_value:
            best_value = board_value
            best_move = move

    end_time = time.time()
    print(f"Time taken to find the best move: {end_time - start_time:.2f} seconds")
    print("Positions evaluated:", position_counts[0])
    return best_move