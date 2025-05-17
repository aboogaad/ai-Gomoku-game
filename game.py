import numpy as np
import pygame
import sys
import time
from pygame.locals import *


# Logic el game
class Gomoku:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.move_history = []

    def reset(self):
        self.__init__(self.size)

    def make_move(self, row, col):
        if self.game_over or not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self.move_history.append((row, col, self.current_player))

        if self.check_winner(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
            self.winner = 0  # Draw

        self.current_player = 3 - self.current_player
        return True

    def undo_move(self):
        if not self.move_history:
            return False

        row, col, player = self.move_history.pop()
        self.board[row][col] = 0
        self.game_over = False
        self.winner = None
        self.current_player = player
        self.last_move = self.move_history[-1][:2] if self.move_history else None
        return True

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def is_board_full(self):
        return np.all(self.board != 0)

    def check_winner(self, row, col):
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                r, c = row + direction * dr, col + direction * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                    r += direction * dr
                    c += direction * dc
            if count >= 5:
                return True
        return False

    def get_valid_moves(self):
        if not self.move_history:
            center = self.size // 2
            return [(center, center)]

        empty_positions = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]

        if len(empty_positions) > 40:
            focused_moves = []
            for i, j in empty_positions:
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] != 0:
                            focused_moves.append((i, j))
                            break
            if focused_moves:
                return focused_moves
        return empty_positions


#El AI player
class AIPlayer:
    def __init__(self, player, algorithm="alphabeta", depth=2):
        self.player = player
        self.algorithm = algorithm
        self.depth = depth
        self.pattern_weights = {
            'five': 100000, 'open_four': 10000, 'four': 1000, 'open_three': 500, 'three': 100, 'open_two': 50, 'two': 10,
            'opp_five': -100000, 'opp_open_four': -10000, 'opp_four': -1000, 'opp_open_three': -500,
            'opp_three': -100, 'opp_open_two': -50, 'opp_two': -10
        }
        self.eval_cache = {}

    def get_move(self, game):
        self.eval_cache = {}
        _, move = (self.minimax(game, self.depth, True) if self.algorithm == "minimax"
                   else self.alphabeta(game, self.depth, -float('inf'), float('inf'), True))
        return move or next(((i, j) for i in range(game.size) for j in range(game.size) if game.board[i][j] == 0), None)

    def minimax(self, game, depth, maximizing):
        if depth == 0 or game.game_over:
            return self.evaluate(game), None
        valid_moves = game.get_valid_moves()
        if maximizing:
            max_eval, best_move = -float('inf'), None
            for move in valid_moves:
                game.make_move(*move)
                eval_score, _ = self.minimax(game, depth - 1, False)
                game.undo_move()
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
            return max_eval, best_move
        else:
            min_eval, best_move = float('inf'), None
            for move in valid_moves:
                game.make_move(*move)
                eval_score, _ = self.minimax(game, depth - 1, True)
                game.undo_move()
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
            return min_eval, best_move

    def alphabeta(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.game_over:
            return self.evaluate(game), None
        valid_moves = game.get_valid_moves()
        if maximizing:
            max_eval, best_move = -float('inf'), None
            for move in valid_moves:
                game.make_move(*move)
                eval_score, _ = self.alphabeta(game, depth - 1, alpha, beta, False)
                game.undo_move()
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval, best_move = float('inf'), None
            for move in valid_moves:
                game.make_move(*move)
                eval_score, _ = self.alphabeta(game, depth - 1, alpha, beta, True)
                game.undo_move()
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, game):
        if game.game_over:
            if game.winner == self.player:
                return 100000
            elif game.winner == 3 - self.player:
                return -100000
            return 0
        board_hash = hash(game.board.tobytes())
        if board_hash in self.eval_cache:
            return self.eval_cache[board_hash]

        score = 0
        for i in range(game.size):
            score += self.evaluate_line(game.board[i, :])
            score += self.evaluate_line(game.board[:, i])
        for i in range(-game.size + 1, game.size):
            score += self.evaluate_line(np.diag(game.board, i))
            score += self.evaluate_line(np.diag(np.fliplr(game.board), i))

        self.eval_cache[board_hash] = score
        return score

    def evaluate_line(self, line):
        line = ''.join(map(str, line))
        p, e, op = str(self.player), '0', str(3 - self.player)
        score = 0
        patterns = {
            'five': p * 5, 'open_four': e + p * 4 + e, 'four': p * 4, 'open_three': e + p * 3 + e,
            'three': p * 3, 'open_two': e + p * 2 + e, 'two': p * 2,
            'opp_five': op * 5, 'opp_open_four': e + op * 4 + e, 'opp_four': op * 4,
            'opp_open_three': e + op * 3 + e, 'opp_three': op * 3, 'opp_open_two': e + op * 2 + e,
            'opp_two': op * 2
        }
        for key, pattern in patterns.items():
            score += line.count(pattern) * self.pattern_weights[key]
        return score


# GUI
class GomokuGUI:
    def __init__(self):
        pygame.init()
        self.board_size = 15
        self.cell_size = 40
        self.margin = 50
        self.width = self.board_size * self.cell_size + 2 * self.margin
        self.height = self.board_size * self.cell_size + 2 * self.margin + 100
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku (Five in a Row)")
        self.colors = {
            'bg': (222, 184, 135), 'line': (0, 0, 0),
            'black': (0, 0, 0), 'white': (255, 255, 255),
            'highlight': (255, 0, 0), 'text': (0, 0, 0),
            'button': (70, 130, 180), 'hover': (100, 149, 237), 'button_text': (255, 255, 255)
        }
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 20)

        self.game = Gomoku(self.board_size)
        self.ai_player = None
        self.ai_player2 = None
        self.ai_thinking = False
        self.game_mode = None
        self.show_menu = True
        self.hovered_button = None

        btn_w, btn_h, margin = 200, 50, 20
        self.buttons = {
            'human_vs_ai': pygame.Rect((self.width - btn_w) // 2, self.height // 2 - btn_h - margin, btn_w, btn_h),
            'ai_vs_ai': pygame.Rect((self.width - btn_w) // 2, self.height // 2, btn_w, btn_h),
            'human_vs_human': pygame.Rect((self.width - btn_w) // 2, self.height // 2 + btn_h + margin, btn_w, btn_h),
            'new_game': pygame.Rect(self.margin, self.height - 70, btn_w // 2, btn_h // 1.5),
            'undo': pygame.Rect(self.margin + btn_w // 2 + 10, self.height - 70, btn_w // 2, btn_h // 1.5),
            'main_menu': pygame.Rect(self.width - self.margin - btn_w, self.height - 70, btn_w, btn_h // 1.5)
        }

    def draw_button(self, rect, text):
        color = self.colors['hover'] if self.hovered_button == rect else self.colors['button']
        pygame.draw.rect(self.window, color, rect, border_radius=5)
        label = self.small_font.render(text, True, self.colors['button_text'])
        self.window.blit(label, label.get_rect(center=rect.center))

    def draw_menu(self):
        self.window.fill(self.colors['bg'])
        title = self.font.render("Gomoku (Five in a Row)", True, self.colors['text'])
        self.window.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 4))
        self.draw_button(self.buttons['human_vs_ai'], "Human vs AI")
        self.draw_button(self.buttons['ai_vs_ai'], "AI vs AI")
        self.draw_button(self.buttons['human_vs_human'], "Human vs Human")

    def start_game(self):
        self.show_menu = False
        self.game.reset()
        if self.game_mode == 1:  # Human vs AI
            self.ai_player = AIPlayer(2, "alphabeta", 2)  # AI is always player 2
            # Human is always player 1 and starts first
        elif self.game_mode == 2:  # AI vs AI
            self.ai_player = AIPlayer(2, "alphabeta", 2)
            self.ai_player2 = AIPlayer(1, "minimax", 2)
            self.ai_thinking = True

    def update(self):
        if self.ai_thinking and not self.game.game_over:
            pygame.display.update()
            pygame.time.delay(300)
            if self.game_mode == 1:
                move = self.ai_player.get_move(self.game)
                self.game.make_move(*move)
                self.ai_thinking = False
            elif self.game_mode == 2:
                current_ai = self.ai_player2 if self.game.current_player == 1 else self.ai_player
                move = current_ai.get_move(self.game)
                self.game.make_move(*move)
                self.ai_thinking = not self.game.game_over

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.hovered_button = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEMOTION:
                    for name, rect in self.buttons.items():
                        if rect.collidepoint(event.pos):
                            self.hovered_button = rect
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            self.window.fill(self.colors['bg'])
            if self.show_menu:
                self.draw_menu()
            else:
                self.draw_board()
                self.update()
            pygame.display.update()
            clock.tick(30)

    def draw_board(self):

        pygame.draw.rect(self.window, self.colors['bg'], (0, 0, self.width, self.height))


        for i in range(self.board_size):

            pygame.draw.line(self.window, self.colors['line'],
                             (self.margin, self.margin + i * self.cell_size),
                             (self.width - self.margin, self.margin + i * self.cell_size), 2)

            pygame.draw.line(self.window, self.colors['line'],
                             (self.margin + i * self.cell_size, self.margin),
                             (self.margin + i * self.cell_size, self.height - self.margin - 100), 2)


        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.board[i][j] == 1:
                    pygame.draw.circle(self.window, self.colors['black'],
                                       (self.margin + j * self.cell_size, self.margin + i * self.cell_size),
                                       self.cell_size // 2 - 2)
                elif self.game.board[i][j] == 2:
                    pygame.draw.circle(self.window, self.colors['white'],
                                       (self.margin + j * self.cell_size, self.margin + i * self.cell_size),
                                       self.cell_size // 2 - 2)


        if self.game.last_move:
            row, col = self.game.last_move
            pygame.draw.circle(self.window, self.colors['highlight'],
                               (self.margin + col * self.cell_size, self.margin + row * self.cell_size),
                               self.cell_size // 6, 2)


        status_text = f"Current: {'Black' if self.game.current_player == 1 else 'White'}"
        if self.game.game_over:
            if self.game.winner == 0:
                status_text = "Game Over: Draw!"
            else:
                status_text = f"Game Over: {'Black' if self.game.winner == 1 else 'White'} wins!"

        status_label = self.font.render(status_text, True, self.colors['text'])
        self.window.blit(status_label, (self.margin, self.height - 120))


        self.draw_button(self.buttons['new_game'], "New Game")
        #self.draw_button(self.buttons['undo'], "Undo")
        self.draw_button(self.buttons['main_menu'], "Main Menu")

    def handle_click(self, pos):
        if self.show_menu:
            if self.buttons['human_vs_ai'].collidepoint(pos):
                self.game_mode = 1  # Human vs AI
                self.start_game()
            elif self.buttons['ai_vs_ai'].collidepoint(pos):
                self.game_mode = 2  # AI vs AI
                self.start_game()
            elif self.buttons['human_vs_human'].collidepoint(pos):
                self.game_mode = 0  # Human vs Human
                self.start_game()
        else:

            if self.buttons['new_game'].collidepoint(pos):
                self.start_game()
            elif self.buttons['undo'].collidepoint(pos) and self.game_mode != 2:
                self.game.undo_move()
            elif self.buttons['main_menu'].collidepoint(pos):
                self.show_menu = True
                return

            if not self.ai_thinking and self.game_mode in (0, 1) and not self.game.game_over:
                if (self.game_mode == 0) or (self.game_mode == 1 and self.game.current_player == 1):
                    x, y = pos[0] - self.margin, pos[1] - self.margin
                    if 0 <= x < self.board_size * self.cell_size and 0 <= y < self.board_size * self.cell_size:
                        col, row = round(x / self.cell_size), round(y / self.cell_size)
                        if self.game.is_valid_move(row, col):
                            self.game.make_move(row, col)
                            if self.game_mode == 1 and not self.game.game_over:
                                self.ai_thinking = True
if __name__ == "__main__":
    gui = GomokuGUI()
    gui.run()
