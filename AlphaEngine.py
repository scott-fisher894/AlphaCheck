import chess
import time

pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900
}

def get_piece_position_value(piece, square):
    """Get the position value of a piece at a given square."""
    if piece.piece_type == chess.PAWN:
        return pst['P'][square]
    elif piece.piece_type == chess.KNIGHT:
        return pst['N'][square]
    elif piece.piece_type == chess.BISHOP:
        return pst['B'][square]
    elif piece.piece_type == chess.ROOK:
        return pst['R'][square]
    elif piece.piece_type == chess.QUEEN:
        return pst['Q'][square]
    elif piece.piece_type == chess.KING:
        return pst['K'][square]
    return 0

def evaluate(board):
    total_value = 0
    # Define the move count after which the AI should play more aggressively
    move_count_threshold = 15  # You can adjust this threshold as needed
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type_value = PIECE_VALUES.get(piece.piece_type, 0)
            piece_position_value = get_piece_position_value(piece, square)
            piece_value = piece_type_value + piece_position_value
            total_value += piece_value if piece.color == chess.WHITE else -piece_value

    # Check for endgame phase and encourage more aggressive play
    if len(board.move_stack) > move_count_threshold:
        if board.is_checkmate():
            # Assign a high positive value if AI can checkmate the opponent
            total_value += 10000
        elif board.is_check():
            # Assign some value for putting the opponent in check
            total_value += 500

    return -total_value if board.turn == chess.BLACK else total_value


def alphabeta(board, depth, alpha, beta, maximizing_player, position_counts):
    if depth == 0 or board.is_game_over():
        position_counts[0] += 1
        return evaluate(board)

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