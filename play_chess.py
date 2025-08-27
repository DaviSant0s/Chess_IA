import torch.nn.functional as F
import torch
import chess
import chess.engine
from treinamento import ChessClassifier, fen_para_tensor, label_to_idx
import json

def evaluate_move(fen, move_uci, model, device):
    """Avalia a qualidade de uma jogada com a CNN."""
    model.eval()
    tensor = fen_para_tensor(fen, move_uci).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        predicted_label = probs.argmax(dim=1).item()
    labels = {0: "boa", 1: "imprecisa", 2: "erro", 3: "blunder"}
    return labels[predicted_label], probs[0, 0].item()

def suggest_move(fen, engine, time_limit=0.1):
    """Sugere uma jogada usando o Stockfish (temporário)."""
    board = chess.Board(fen)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move.uci()

def main():
    # Configuração
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega o modelo treinado
    model = ChessClassifier().to(device)
    model.load_state_dict(torch.load("modelo_chessia.pt"))

    # Configura o Stockfish (para sugestões temporárias)
    STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Inicializa o tabuleiro
    board = chess.Board()
    print("Bem-vindo ao jogo de xadrez! Digite jogadas em notação UCI (ex.: e2e4).")
    print("Digite 'sair' para encerrar.")

    def print_board_with_coords():
        board_str = str(board).split("\n")
        for i, row in enumerate(board_str):
            print(f"{8-i} {row}")  # numeração à esquerda
        print("  a b c d e f g h")  # letras embaixo

    while not board.is_game_over():
        # Exibe o tabuleiro
        print_board_with_coords()
        print(f"Turno: {'Brancas' if board.turn == chess.WHITE else 'Pretas'}")

        # Sugere uma jogada (usando Stockfish temporariamente)
        suggestion = suggest_move(board.fen(), engine)
        print(f"Sugestão de jogada: {suggestion}")

        # Recebe a jogada do jogador
        move_uci = input("Digite sua jogada (UCI): ")
        if move_uci.lower() == "sair":
            break

        # Valida a jogada
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                print("Jogada inválida! Tente novamente.")
                continue
        except:
            print("Formato UCI inválido! Exemplo: e2e4")
            continue

        # Avalia a jogada com a CNN
        fen_before = board.fen()
        label, prob_boa = evaluate_move(fen_before, move_uci, model, device)
        print(f"Avaliação da jogada: {label} (Prob. de 'boa': {prob_boa:.2%})")

        # Aplica a jogada
        board.push(move)

    # Exibe o resultado do jogo
    if board.is_game_over():
        result = board.result()
        print(f"Jogo terminado! Resultado: {result}")
    engine.quit()

if __name__ == "__main__":
    main()
