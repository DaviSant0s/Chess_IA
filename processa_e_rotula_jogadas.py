import chess.pgn
import json
import chess.engine

# Etapa 1: Processar jogadas do PGN
def processamento_pgn():
    jogadas_para_treino = []

    with open("lichess_db_standard_rated_2013-01.pgn", encoding="utf-8") as pgn_file:

        contador = 0
        LIMITE = 5

        while contador < LIMITE:
            partida = chess.pgn.read_game(pgn_file)
            if partida is None:
                break

            tabuleiro = partida.board()

            for lance in partida.mainline_moves():
                estado_fen_antes = tabuleiro.fen()
                # jogada = tabuleiro.san(lance)
                tabuleiro.push(lance)

                jogadas_para_treino.append({
                    "fen_antes": estado_fen_antes,
                    "jogada": lance.uci(),
                    "label": None
                })

            contador += 1

    # Salva as jogadas em um arquivo json de forma identada
    with open("jogadas_para_treino.json", "w", encoding="utf-8") as f:
        json.dump(jogadas_para_treino, f, indent=2)


# Etapa 2: Avaliação com Stockfish (preenchimento dos labels)

# Função para converter score em centipawns
def score_to_cp(score):
    if score.is_mate():
        return 100000 if score.mate() > 0 else -100000
    else:
        return score.score()

# Define faixas para classificação da jogada
def classificar_jogada(dif_cp):
    dif_cp = abs(dif_cp)
    if dif_cp < 50:
        return "boa"
    elif dif_cp < 100:
        return "imprecisa"
    elif dif_cp < 300:
        return "erro"
    else:
        return "blunder"
    
def rotula_jogadas_stockfish():

    # Caminho para o executável do Stockfish
    STOCKFISH_PATH = "./stockfish-ubuntu-x86-64-avx2"
    
    # Abrir o arquivo das jogadas preparadas
    with open("jogadas_para_treino.json", "r", encoding="utf-8") as f:
        jogadas = json.load(f)

    # Inicializar o motor Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    ANALISE_TIME = 0.1
    jogadas_rotuladas = []

    for i, item in enumerate(jogadas):
        fen = item["fen_antes"]
        jogada_uci = item["jogada"]
        board = chess.Board(fen)

        info_antes = engine.analyse(board, chess.engine.Limit(time=ANALISE_TIME))

        # clacula o score anterior
        score_antes = info_antes["score"].white()

        move = chess.Move.from_uci(jogada_uci)
        board.push(move)

        info_depois = engine.analyse(board, chess.engine.Limit(time=ANALISE_TIME))

        # clacula o score depois
        score_depois = info_depois["score"].white()

        diff = score_to_cp(score_antes) - score_to_cp(score_depois)
        label = classificar_jogada(diff)

        item["label"] = label
        jogadas_rotuladas.append(item)

        if i % 50 == 0:
            print(f"Avaliado {i+1}/{len(jogadas)} jogadas...")

    engine.quit()

    with open("jogadas_rotuladas.json", "w", encoding="utf-8") as f:
        json.dump(jogadas_rotuladas, f, indent=2)

    print("Rotulação concluída!")