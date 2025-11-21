import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# Mapeamento de rótulos
label_to_idx = {"boa": 0, "imprecisa": 1, "erro": 2, "blunder": 3}

# Converte FEN e jogada para tensor 14x8x8
def fen_para_tensor(fen, jogada):
    board = chess.Board(fen)
    plano = torch.zeros(14, 8, 8, dtype=torch.float32)  # 12 peças + 2 jogada

    pecas = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }

    # Preenche os 12 canais de peças
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7 - i))
            if piece:
                idx = pecas[piece.symbol()]
                plano[idx, i, j] = 1.0

    # Codifica a jogada
    move = chess.Move.from_uci(jogada)
    origem = move.from_square
    destino = move.to_square
    origem_i, origem_j = 7 - (origem // 8), origem % 8
    destino_i, destino_j = 7 - (destino // 8), destino % 8
    plano[12, origem_i, origem_j] = 1.0  # Casa de origem
    plano[13, destino_i, destino_j] = 1.0  # Casa de destino

    return plano

# Modelo CNN simples
class ChessClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # self.pool = nn.MaxPool2d(2, 2)  # reduz pela metade a dimensão espacial

        self.dropout = nn.Dropout(0.3)

        # Como usamos 3 pools, a grade 8x8 vira 1x1 → 256 canais finais
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # saída para 4 classes

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)