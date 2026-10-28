"""Competitive environments with shared action-policy RL interfaces."""

ENVIRONMENTS = ('market', 'maze', 'chess')


def make_env(name, seed, regime='train'):
    if name == 'market':
        from .market import MarketEnv
        return MarketEnv(seed, regime=regime, max_rounds=6)
    if name == 'maze':
        from .maze import MazeEnv
        return MazeEnv(seed, regime=regime, max_rounds=8)
    if name == 'chess':
        from .chess_game import ChessEnv
        return ChessEnv(seed, regime=regime, max_rounds=16)
    raise ValueError('Unknown environment: ' + name)
