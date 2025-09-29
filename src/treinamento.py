import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import chess
import json
import os
import glob
import time

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
    print(f"[INFO] Usando dispositivo: {device}")

    # Carrega o dataset completo
    dataset = ChessDataset("jogadas_rotuladas.json")

    # Divide entre treino (80%) e teste (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"[INFO] Dataset carregado: {len(dataset)} jogadas")
    print(f"[INFO] Treino: {len(train_dataset)} | Teste: {len(test_dataset)}")

    # Cria DataLoaders para carregar dados em lotes de 32 exemplos.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ChessClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 3
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========== Carregar último checkpoint se existir ===================
    start_epoch = 0

    # Variável para controlar o melhor modelo
    melhor_acc = 0.0
    melhor_model_path = os.path.join(checkpoint_dir, "melhor_modelo.pt")

    # pega todos os arquivos que terminam em .pt
    # O Sort() organiza por nome → assim o último da lista será o mais recente.
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "modelo_epoch*.pt")))

    # Se existir pelo menos 1 checkpoint, carrega o mais recente.
    if checkpoint_files:
        # Pega o último checkpoint da lista.
        last_ckpt = checkpoint_files[-1]

        # Carrega o conteúdo do checkpoint
        checkpoint = torch.load(last_ckpt, map_location=device)

        # Atualiza os pesos da rede neural com os valores salvos no checkpoint.
        # Faz o modelo continuar de onde parou, em vez de reiniciar do zero.
        model.load_state_dict(checkpoint["model_state"])

        # Recupera o estado interno do otimizador
        # Isso é importante porque garante que o aprendizado continue suave, sem resetar a curva de treino.
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Lê a época (epoch) salva no checkpoint e soma +1.
        # Assim, o próximo treino começa na época seguinte e não repete a última.
        start_epoch = checkpoint["epoch"] + 1

        # Confirma qual checkpoint foi carregado e de qual época o treino vai retomar
        print(f"[INFO] Checkpoint carregado: {last_ckpt} | Retomando da epoch {start_epoch+1}")

        # === Atualiza melhor_acc se existir melhor_modelo.pt ===
        if os.path.exists(melhor_model_path):
            # Carrega os pesos do melhor modelo e avalia no conjunto de teste para obter a acurácia

            model.load_state_dict(torch.load(melhor_model_path, map_location=device))
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    correct += (outputs.argmax(1) == y).sum().item()
                    total += y.size(0)
            melhor_acc = (correct / total) * 100
            print(f"[INFO] Melhor modelo carregado: {melhor_model_path} | Acc: {melhor_acc:.2f}%")
        
        # Recarrega o último checkpoint para continuar o treino normalmente
        model.load_state_dict(checkpoint["model_state"])

    # Loop de Treinamento
    for epoch in range(start_epoch, num_epochs):
        inicio = time.time()
        model.train() # Ativa o modo de treinamento
        total_loss = 0
        correct = 0
        total = 0

        inicio_epoch = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):

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

            
            # O print só acontece a cada 1000 batches → evita inundar o terminal.
            # Ele imprime: Época atual, Lote atual / total, Loss do batch atual, Acurácia parcial até agora
            if (batch_idx + 1) % 1000 == 0:
                acc_parcial = (correct / total) * 100

                tempo_passado = time.time() - inicio_epoch
                
                print(f"[TREINO] Epoch {epoch+1}/{num_epochs} | Lote {batch_idx+1}/{len(train_loader)} | "
                    f"Loss parcial: {loss.item():.4f} | Acc parcial: {acc_parcial:.2f}% | "
                    f"Tempo passado: {tempo_passado:.2f}s")

        # Calcula e imprime a acurácia de treinamento (acc) e a perda média por época.
        acc = (correct / total) * 100
        fim = time.time()
        print(f"[EPOCH {epoch+1}] Loss médio: {total_loss/len(train_loader):.4f} | "
              f"Acurácia treino: {acc:.2f}% | Tempo: {fim - inicio:.2f}s")

        # Avaliação no conjunto de teste
        model.eval() # Ativa o modo de avaliação (desativa dropout).
        correct = 0
        total = 0
        test_loss = 0

        # Desativa o cálculo de gradientes para economizar memória.
        with torch.no_grad():

            # Para cada lote no test_loader
            for x, y in test_loader:

                # Move x e y para o dispositivo.
                x = x.to(device) 
                y = y.to(device)

                # Passa x pela CNN para obter outputs
                outputs = model(x)

                # Calcula loss em cada batch de teste e acumula em test_loss.
                loss = criterion(outputs, y)
                test_loss += loss.item()

                # Conta previsões corretas comparando outputs.argmax(1) (classe prevista) com y (classe verdadeira).
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        # Calcula e imprime a acurácia no conjunto de teste.
        test_acc = (correct / total) * 100

        # Calcula acurácia final do teste.
        # Mostra também loss médio no conjunto de teste, dividido pelo número de batches
        print(f"[TESTE] Epoch {epoch+1} | Loss: {test_loss/len(test_loader):.4f} | " f"Acurácia: {test_acc:.2f}%")

        # Log adicional de conclusão da época com melhor acurácia até agora
        print(f"[INFO] Epoch {epoch+1} concluída, melhor acc: {melhor_acc:.2f}%")

        # === Salvar checkpoint ===

        # Ao final de cada época, salva: 
        # número da época (epoch), pesos da rede (model_state), estado do otimizador (optimizer_state)
        checkpoint_path = os.path.join(checkpoint_dir, f"modelo_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)
        print(f"[INFO] Checkpoint salvo: {checkpoint_path}")

        # === Salvar o melhor modelo com base na acurácia de teste ===

        # Se a acurácia dessa época for maior que a melhor anterior, significa que o modelo dessa época é o melhor até agora.
        if test_acc > melhor_acc:

            # guarda a melhor acurácia
            melhor_acc = test_acc

            # Salva apenas os pesos do modelo (state_dict) no arquivo definido em melhor_model_path.
            # Esse arquivo sempre conterá o modelo que teve a melhor acurácia até o momento
            torch.save(model.state_dict(), melhor_model_path)

            print(f"[INFO] Novo melhor modelo salvo: {melhor_model_path} | Acc: {melhor_acc:.2f}%")

    print("[INFO] Treinamento concluído.")

# Executa o treinamento
if __name__ == "__main__":
    treinar_modelo()