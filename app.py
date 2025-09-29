from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import chess
import chess.engine
from treinamento import ChessClassifier, fen_para_tensor, label_to_idx
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite todas as origens

# Configuração do dispositivo e modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessClassifier().to(device)
model.load_state_dict(torch.load("./checkpoints/melhor_modelo.pt", map_location=device))
model.eval()

# Stockfish
STOCKFISH_PATH = "./scripts/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Tabuleiro global (para simplificação, um jogo por vez)
board = chess.Board()

# Função para avaliar jogada
def evaluate_move(fen, move_uci):
    tensor = fen_para_tensor(fen, move_uci).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        predicted_label = probs.argmax(dim=1).item()
    labels = {0: "boa", 1: "imprecisa", 2: "erro", 3: "blunder"}
    return labels[predicted_label], probs[0, 0].item()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bem-vindo! A API de ChessIA está online e pronta para usar."})

# Endpoint para retornar o estado atual do tabuleiro e de quem é a vez
@app.route("/state", methods=["GET"])
def state():
    fen = board.fen()
    turn = "white" if board.turn else "black"
    return jsonify({
        "fen": fen,
        "turn": turn
    })

# Endpoint para sugerir jogada
@app.route("/suggest", methods=["GET"])
def suggest():
    fen = board.fen()
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return jsonify({"suggestion": result.move.uci(), "fen": fen})

# Endpoint para avaliar jogada
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    move_uci = data.get("move")
    fen = board.fen()
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"error": "Jogada inválida"}), 400
    except:
        return jsonify({"error": "Formato UCI inválido"}), 400

    label, prob = evaluate_move(fen, move_uci)
    return jsonify({"evaluation": label, "probability": prob, "fen": fen})


# Endpoint para jogar a jogada no tabuleiro
@app.route("/move", methods=["POST"])
def move():
    data = request.json
    move_uci = data.get("move")
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"error": "Jogada inválida"}), 400
    except:
        return jsonify({"error": "Formato UCI inválido"}), 400

    label, prob = evaluate_move(board.fen(), move_uci)

    board.push(move)

    return jsonify({"fen": board.fen(), "result": board.result() if board.is_game_over() else None, "label": label, "prob": prob})


# Resetar o jogo
@app.route("/reset", methods=["POST"])
def reset():
    global board
    board = chess.Board()
    return jsonify({"message": "Jogo reiniciado", "fen": board.fen()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)