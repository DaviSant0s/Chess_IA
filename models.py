from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
import chess

# 1. Crie as instâncias das extensões AQUI, mas sem o (app)
# Elas ficam "vazias" por enquanto.
db = SQLAlchemy()
bcrypt = Bcrypt()

# 2. Cole suas classes de modelo exatamente como você fez
class User(db.Model):
    """
    Tabela de Usuários. Armazena informações de login e perfil.
    """
    __tablename__ = "user"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    rating = db.Column(db.Integer, nullable=False, default=1200)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    games_as_white = db.relationship('Game', foreign_keys='Game.player_white_id', back_populates='player_white')
    games_as_black = db.relationship('Game', foreign_keys='Game.player_black_id', back_populates='player_black')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Game(db.Model):
    """
    Tabela de Partidas. Armazena APENAS o estado ATUAL de um jogo.
    """
    __tablename__ = "game"
    
    game_id = db.Column(db.String(8), primary_key=True)

    player_white_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    player_black_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    status = db.Column(db.String(20), nullable=False, default='waiting')

    result = db.Column(db.String(10), nullable=True)
    current_fen = db.Column(db.String(128), nullable=False, default=chess.Board().fen())
    
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_move_at = db.Column(db.DateTime, nullable=True, onupdate=datetime.utcnow)
    
    player_white = db.relationship('User', foreign_keys=[player_white_id], back_populates='games_as_white')
    player_black = db.relationship('User', foreign_keys=[player_black_id], back_populates='games_as_black')
    
    def __repr__(self):
        return f'<Game {self.game_id} (Status: {self.status})>'