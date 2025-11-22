import os
import uuid
import chess
import chess.engine
import torch
import torch.nn.functional as F
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. IMPORTA AS FUNÇÕES DO SOCKETIO
from flask_socketio import SocketIO, join_room, leave_room, emit

from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity
)

from models import db, bcrypt, User, Game

# Importa SUAS FUNÇÕES DE IA
try:
    from treinamento import ChessClassifier, fen_para_tensor
except ImportError:
    print("AVISO: Arquivo 'treinamento.py' não encontrado. A avaliação de IA falhará.")
    class ChessClassifier: pass
    def fen_para_tensor(fen, move): return None
    def label_to_idx(label): return 0

# --- CONFIGURAÇÃO DO APP ---

app = Flask(__name__)
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))

# Configurações Essenciais
app.config['SECRET_KEY'] = 'minha-chave-secreta-muito-segura-mude-depois'
app.config['JWT_SECRET_KEY'] = 'minha-chave-jwt-segura-mude-tambem'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'games.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- INICIALIZAÇÃO DAS EXTENSÕES ---
db.init_app(app)
bcrypt.init_app(app)

# 2. INICIALIZE O SOCKETIO
# Coloque o modo async='threading' para funcionar bem com o app.run()
socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173", async_mode='threading')

jwt = JWTManager(app)

# --- CARREGAMENTO DOS MODELOS DE IA E STOCKFISH ---
print("Carregando modelos de IA e Stockfish...")
device = torch.device("cpu")
print(f"Usando dispositivo: {device}")
model = ChessClassifier().to(device)
try:
    model.load_state_dict(torch.load("./melhor_modelo.pt", map_location=device))
    model.eval()
    print("Modelo de IA (melhor_modelo.pt) carregado com sucesso.")
except Exception as e:
    print(f"AVISO: Falha ao carregar modelo de IA. Erro: {e}")

STOCKFISH_PATH = "./scripts/stockfish-ubuntu-x86-64-avx2"
engine = None
try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print("Motor Stockfish carregado com sucesso.")
except Exception as e:
    print(f"AVISO: Falha ao carregar Stockfish. Erro: {e}")


# --- FUNÇÕES AUXILIARES (IA e JOGO) ---

def evaluate_move(fen, move_uci):
    """Sua função original para avaliar uma jogada com o modelo PyTorch."""
    try:
        tensor = fen_para_tensor(fen, move_uci).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            predicted_label = probs.argmax(dim=1).item()
        labels = {0: "boa", 1: "imprecisa", 2: "erro", 3: "blunder"}
        return labels[predicted_label], probs[0, 0].item()
    except Exception as e:
        print(f"ERRO ao avaliar jogada: {e}")
        return "desconhecida", 0.0

def get_game_result(board):
    """Verifica se o jogo acabou e retorna o status."""
    if board.is_checkmate():
        return "checkmate", board.result()
    if board.is_stalemate():
        return "stalemate", "1/2-1/2"
    if board.is_insufficient_material():
        return "draw", "1/2-1/2"
    if board.can_claim_draw():
        return "draw", "1/2-1/2"
    return "ongoing", None

# --- ROTAS DE AUTENTICAÇÃO E JOGO (HTTP) ---

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    if not all([username, email, password]):
        return jsonify({"error": "Todos os campos são obrigatórios"}), 400
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({"error": "Usuário ou e-mail já existe"}), 400
    new_user = User(username=username, email=email)
    new_user.set_password(password)
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": f"Usuário {username} criado com sucesso!"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=str(user.id))
        return jsonify(access_token=access_token)
    return jsonify({"error": "Usuário ou senha inválidos"}), 401

@app.route("/profile", methods=["GET"])
@jwt_required() 
def profile():
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    return jsonify(username=user.username, email=user.email, rating=user.rating), 200

@app.route("/create_game", methods=["POST"])
@jwt_required()
def create_game():
    current_user_id = int(get_jwt_identity())
    data = request.json
    play_as = data.get("play_as", "white") 
    game_id = str(uuid.uuid4().hex)[:8] 
    new_game = Game(
        game_id=game_id,
        current_fen=chess.Board().fen()
    )
    if play_as == "white":
        new_game.player_white_id = current_user_id
    else:
        new_game.player_black_id = current_user_id
    try:
        db.session.add(new_game)
        db.session.commit()
        return jsonify({
            "message": "Jogo criado!",
            "game_id": new_game.game_id,
            "fen": new_game.current_fen
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/games", methods=["GET"])
def get_open_games():
    open_games = Game.query.filter(
        Game.status == 'waiting',
        (Game.player_white_id == None) | (Game.player_black_id == None)
    ).all()
    games_list = [{
        "game_id": g.game_id,
        "needs_player": "white" if g.player_white_id is None else "black",
        "created_at": g.created_at
    } for g in open_games]
    return jsonify(games_list), 200

# --- ROTA DE IA (HTTP) ---

@app.route("/suggest", methods=["POST"])
@jwt_required() 
def suggest():
    game_id = request.json.get("game_id")
    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
    if not engine:
         return jsonify({"error": "Motor Stockfish não inicializado no servidor"}), 500
    board = chess.Board(game.current_fen)
    if board.is_game_over():
        return jsonify({"error": "O jogo já acabou"}), 400
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return jsonify({"suggestion": result.move.uci(), "fen": game.current_fen})


# --- NOVAS ROTAS DE JOGO (HTTP) ---

def get_full_game_state(game: Game):
    """Função auxiliar para montar o payload de resposta do estado do jogo."""
    board = chess.Board(game.current_fen)
    return {
        'game_id': game.game_id,
        'fen': game.current_fen,
        'status': game.status,
        'result': game.result,
        'turn': 'white' if board.turn else 'black',
        'player_white': game.player_white.username if game.player_white else None,
        'player_black': game.player_black.username if game.player_black else None,
        'last_move_at': game.last_move_at.isoformat() if game.last_move_at else None,
        # --- INÍCIO DA ADIÇÃO ---
        'rematch_requested_by': game.rematch_requested_by.username if game.rematch_requested_by else None
        # --- FIM DA ADIÇÃO ---
    }

@app.route("/game_state/<game_id>", methods=["GET"])
@jwt_required() # Protegido, para que só jogadores logados vejam os jogos
def get_game_state(game_id):
    """
    Esta é a rota de POLLING. 
    O frontend deve chamar esta rota a cada X segundos para verificar atualizações.
    """
    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
        
    # (Opcional) Você pode adicionar uma verificação se o usuário atual
    # tem permissão para ver este jogo.
    # current_user_id = get_jwt_identity()
    # if current_user_id not in [game.player_white_id, game.player_black_id]:
    #     return jsonify({"error": "Não autorizado a ver este jogo"}), 403

    return jsonify(get_full_game_state(game)), 200


@app.route("/join_game", methods=["POST"])
@jwt_required()
def join_game():
    """
    Permite que um usuário logado entre em um jogo que tenha um slot vazio.
    """
    game_id = request.json.get('game_id')
    current_user_id = int(get_jwt_identity())
    
    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
        
    if game.status != "waiting":
        return jsonify({"error": "Este jogo não está esperando jogadores"}), 400

    # Lógica para "entrar" no jogo
    if game.player_white_id is None and current_user_id != game.player_black_id:
        game.player_white_id = current_user_id
        game.status = 'ongoing' # O jogo começa!
    elif game.player_black_id is None and current_user_id != game.player_white_id:
        game.player_black_id = current_user_id
        game.status = 'ongoing' # O jogo começa!
    else:
        return jsonify({"error": "Este jogo já está cheio"}), 400
    
    try:
        db.session.commit()
        
        # 1. Prepara o estado atualizado
        updated_game_state = get_full_game_state(game)

        # 2. 🌟 A CORREÇÃO: Emite para a sala (Notifica o 'davi')
        socketio.emit('game_update', updated_game_state, room=game.game_id)

        # 3. Retorna para o 'joao' (via HTTP)
        return jsonify(updated_game_state), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/move", methods=["POST"])
@jwt_required()
def make_move():
    """
    Recebe uma jogada de um cliente e a executa.
    Substitui o antigo 'handle_make_move' do socket.
    """
    data = request.json
    game_id = data.get('game_id')
    move_uci = data.get('move')
    
    current_user_id = int(get_jwt_identity())

    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
    
    if game.status != "ongoing":
        return jsonify({"error": "Este jogo não está em andamento"}), 400

    board = chess.Board(game.current_fen)
    
    # Validação Crítica da Jogada
    is_white_turn = board.turn == chess.WHITE
    is_black_turn = board.turn == chess.BLACK

    if (is_white_turn and current_user_id != game.player_white_id) or \
       (is_black_turn and current_user_id != game.player_black_id):
        return jsonify({"error": "Não é o seu turno"}), 403
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"error": "Jogada ilegal"}), 400
    except:
        return jsonify({"error": "Formato de jogada (UCI) inválido"}), 400
        
    # AVALIAÇÃO DA IA
    label, prob = evaluate_move(game.current_fen, move_uci)

    # Faz a jogada e verifica o resultado
    board.push(move)
    new_status, result = get_game_result(board)
    
    # Atualiza o banco de dados
    game.current_fen = board.fen()
    game.status = new_status
    game.result = result
    game.last_move_at = datetime.utcnow()
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f'Erro de banco de dados: {str(e)}'}), 500

    # Prepara a resposta
    response_data = get_full_game_state(game)
    response_data['last_move'] = move_uci
    response_data['evaluation'] = {
        'label': label,
        'probability_good': prob
    }

    # 3. A GRANDE MUDANÇA: EMITIR O ESTADO ATUALIZADO
    # Emite para a "sala" com o nome do game_id.
    # Todos os clientes (incluindo o que fez a jogada)
    # que estiverem nessa sala receberão o 'game_update'.
    socketio.emit('game_update', response_data, room=game_id)
    
    # Retorna a resposta para o jogador que fez a jogada
    return jsonify(response_data), 200

# --- INÍCIO DAS NOVAS ROTAS ---

@app.route("/request_rematch", methods=["POST"])
@jwt_required()
def request_rematch():
    game_id = request.json.get('game_id')
    current_user_id = int(get_jwt_identity())
    
    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
    
    if game.status in ['waiting', 'ongoing']:
        return jsonify({"error": "O jogo ainda não acabou"}), 400
    
    if current_user_id not in [game.player_white_id, game.player_black_id]:
        return jsonify({"error": "Você não é um jogador nesta partida"}), 403

    game.rematch_requested_by_id = current_user_id
    
    try:
        db.session.commit()
        # Notifica a todos na sala sobre o pedido
        socketio.emit('game_update', get_full_game_state(game), room=game_id)
        return jsonify(get_full_game_state(game)), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/accept_rematch", methods=["POST"])
@jwt_required()
def accept_rematch():
    game_id = request.json.get('game_id')
    current_user_id = int(get_jwt_identity())
    
    game = Game.query.get(game_id)
    if not game:
        return jsonify({"error": "Jogo não encontrado"}), 404
        
    if not game.rematch_requested_by_id:
        return jsonify({"error": "Nenhum pedido de revanche encontrado"}), 400
            
    if game.rematch_requested_by_id == current_user_id:
        return jsonify({"error": "Você não pode aceitar seu próprio pedido"}), 400

    # Reseta o jogo "no lugar" (in-place)
    game.current_fen = chess.Board().fen()
    game.status = 'ongoing'
    game.result = None
    game.last_move_at = datetime.utcnow()
    game.rematch_requested_by_id = None # Limpa o pedido
    
    # Inverte os jogadores
    old_white_id = game.player_white_id
    old_black_id = game.player_black_id
    
    game.player_white_id = old_black_id
    game.player_black_id = old_white_id
    
    try:
        db.session.commit()
        # Notifica a todos sobre o "novo" jogo
        updated_state = get_full_game_state(game)
        socketio.emit('game_update', updated_state, room=game_id)
        return jsonify(updated_state), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
        
# --- FIM DAS NOVAS ROTAS ---

@socketio.on('join_game')
def handle_join_game(data):
    """
    Chamado pelo cliente quando ele entra na tela de um jogo.
    Coloca o cliente na "sala" daquele jogo.
    """
    game_id = data.get('game_id')
    if game_id:
        join_room(game_id)
        print(f"Cliente {request.sid} entrou na sala: {game_id}")

@socketio.on('leave_game')
def handle_leave_game(data):
    """ (Opcional, mas bom ter)
    Chamado pelo cliente quando ele sai da tela de um jogo.
    """
    game_id = data.get('game_id')
    if game_id:
        leave_room(game_id)
        print(f"Cliente {request.sid} saiu da sala: {game_id}")


# --- BLOCO DE EXECUÇÃO ---

if __name__ == "__main__":
    with app.app_context():
        db.create_all() 
        print("Banco de dados e tabelas criados com sucesso (se não existiam).")
        
    print("Iniciando servidor Flask (sem SocketIO)...")
    # 4. MUDE A FORMA DE INICIAR O SERVIDOR
    # Use socketio.run() ao invés de app.run()
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)