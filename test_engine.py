import chess
import chess.engine
import random
import AlphaEngine as traditional_engine
import AlphaEngine_NN as nn_engine
import torch

def generate_random_board():
    board = chess.Board()
    num_moves = random.randint(5, 6)  # Generate up to 20 moves for randomness

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            break
        move = random.choice(legal_moves)
        board.push(move)

    return board

def main():
    # Load NN model
    model = nn_engine.ChessEvaluationNN()
    model.load_state_dict(torch.load('chess_model.pth'))
    model.eval()

    # Generate a random chess board
    random_board = generate_random_board()
    print("Random Board State:\n", random_board)

    # Traditional Engine Move
    traditional_best_move = traditional_engine.select_best_move(random_board, depth=4)
    print("Traditional Engine Suggests:", traditional_best_move)

    # Neural Network Engine Move
    nn_best_move = nn_engine.select_best_move(random_board, depth=4)
    print("Neural Network Engine Suggests:", nn_best_move)

if __name__ == "__main__":
    main()
