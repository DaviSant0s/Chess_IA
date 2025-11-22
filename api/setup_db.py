# setup_db.py
import os
from app import app, db

# Se você precisar de alguma variável de ambiente específica para o DB, carregue aqui

with app.app_context():
    # Cria todas as tabelas definidas nas models (User, Game)
    db.create_all() 
    print("Tabelas criadas com sucesso durante o deploy.")