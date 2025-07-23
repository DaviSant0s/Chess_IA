import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import chess
import json

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

# Dataset personalizado
class ChessDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = fen_para_tensor(item["fen_antes"], item["jogada"])
        y = label_to_idx[item["label"]]
        return x, y

# Modelo CNN simples
class ChessClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)  # 14 canais
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Função de treino
def treinar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega o dataset completo
    dataset = ChessDataset("jogadas_rotuladas.json")

    # Divide entre treino (80%) e teste (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Cria DataLoaders para carregar dados em lotes de 32 exemplos.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # - Cria uma instância da CNN (ChessClassifier) e a move para o dispositivo (cuda ou cpu).
    # - Usa o otimizador Adam com taxa de aprendizado lr=0.001 para atualizar os pesos do modelo.
    # - Usa CrossEntropyLoss como função de perda, que combina log-softmax e perda de entropia 
    # cruzada para classificação multiclasse (4 classes).

    model = ChessClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    # Loop de Treinamento, Treina o modelo por 10 épocas (num_epochs = 10).
    for epoch in range(num_epochs):
        model.train() # Ativa o modo de treinamento
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad() # Zera os gradientes acumulados.
            outputs = model(x) # Passa o tensor x
            loss = criterion(outputs, y) # Calcula a perda entre os logits previstos e os rótulos verdadeiros (y).
            loss.backward() # Calcula os gradientes da perda em relação aos pesos do modelo.
            optimizer.step() # Atualiza os pesos usando o otimizador Adam.

            total_loss += loss.item() # Acumula a perda
            correct += (outputs.argmax(1) == y).sum().item() # contagem de previsões corretas
            total += y.size(0) # total de exemplos

        # Calcula e imprime a acurácia de treinamento (acc) e a perda média por época.
        acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Acurácia (treino): {acc:.2f}%")

    # Avaliação no conjunto de teste
    model.eval() # Ativa o modo de avaliação (desativa dropout).
    correct = 0
    total = 0

    # Desativa o cálculo de gradientes para economizar memória.
    with torch.no_grad():

        # Para cada lote no test_loader
        for x, y in test_loader:

            # Move x e y para o dispositivo.
            x = x.to(device) 
            y = y.to(device)

            # Passa x pela CNN para obter outputs
            outputs = model(x)

            # Conta previsões corretas comparando outputs.argmax(1) (classe prevista) com y (classe verdadeira).
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    # Calcula e imprime a acurácia no conjunto de teste.
    test_acc = (correct / total) * 100
    print(f"Acurácia no conjunto de teste: {test_acc:.2f}%")

    # Salva o modelo treinado
    torch.save(model.state_dict(), "modelo_chessia.pt")
    print("Modelo salvo como modelo_chessia.pt")

# Executa o treinamento
if __name__ == "__main__":
    treinar_modelo()