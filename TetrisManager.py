from copy import deepcopy
class TetrisSaver:
    def __init__(self):
        self.state = {}

    def save(self, tetris):
        self.state = {
            'grid': deepcopy(tetris.grid),
            'oldko': tetris.oldko,
            '_n_used_block': tetris._n_used_block,
            'buffer': deepcopy(tetris.buffer),
            'held': tetris.held,
            'block': deepcopy(tetris.block),
            'sent': tetris.sent,
            'tempsend': tetris.tempsend,
            'oldcombo': tetris.oldcombo,
            'combo': tetris.combo,
            'tspin': tetris.tspin,
            'now_back2back': tetris.now_back2back,
            'pre_back2back': tetris.pre_back2back,
            'tetris': tetris.tetris,
            '_KO': tetris._KO,
            '_attacked': tetris._attacked,
            '_is_fallen': tetris._is_fallen,
            'px': tetris.px,
            'py': tetris.py,
            'cleared': tetris.cleared,
            'kocounter': tetris.kocounter,
            'stopcounter': tetris.stopcounter,
            'isholded': tetris.isholded,
            'pressedRight': tetris.pressedRight,
            'pressedLeft': tetris.pressedLeft,
            'pressedDown': tetris.pressedDown,
            'LAST_ROTATE_TIME': tetris.LAST_ROTATE_TIME,
            'LAST_MOVE_SHIFT_TIME': tetris.LAST_MOVE_SHIFT_TIME,
            'LAST_MOVE_DOWN_TIME': tetris.LAST_MOVE_DOWN_TIME,
            'LAST_COMBO_DRAW_TIME': tetris.LAST_COMBO_DRAW_TIME,
            'LAST_TETRIS_DRAW_TIME': tetris.LAST_TETRIS_DRAW_TIME,
            'LAST_TSPIN_DRAW_TIME': tetris.LAST_TSPIN_DRAW_TIME,
            'LAST_BACK2BACK_DRAW_TIME': tetris.LAST_BACK2BACK_DRAW_TIME,
            'LAST_NATRUAL_FALL_TIME': tetris.LAST_NATRUAL_FALL_TIME,
            'LAST_FALL_DOWN_TIME': tetris.LAST_FALL_DOWN_TIME,
            'tetris_drawing': tetris.tetris_drawing,
            'tspin_drawing': tetris.tspin_drawing,
            'back2back_drawing': tetris.back2back_drawing,
            'combo_counter': tetris.combo_counter,
            'natural_down_counter': tetris.natural_down_counter
        }

    def load(self, tetris):
        for key, value in self.state.items():
            setattr(tetris, key, value)
