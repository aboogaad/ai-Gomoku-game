# Gomoku (Five in a Row) 🎯

A classic implementation of Gomoku, also known as Five in a Row, built with Python using the `pygame` library. The game includes support for Human vs Human, Human vs AI, and AI vs AI gameplay modes. The AI uses either Minimax or Alpha-Beta Pruning algorithms with pattern-based heuristic evaluation.

---

## 🧠 Game Features

- ✅ **Three Game Modes**:
  - Human vs Human
  - Human vs AI (Alpha-Beta Pruning)
  - AI vs AI (Alpha-Beta vs Minimax)

- 🤖 **AI Logic**:
  - Heuristic evaluation based on pattern matching (e.g., open four, three in a row).
  - Alpha-Beta Pruning and Minimax implemented with move pruning.
  - Caching evaluation scores for better performance.

- 🎮 **Game UI**:
  - Built with `pygame`.
  - Interactive main menu.
  - Visual highlights for last move.
  - Status messages and end-game notifications.
  - Responsive buttons for starting a new game or returning to the main menu.

---

## 📦 Requirements

- Python 3.x
- `pygame`
- `numpy`

You can install the requirements using pip:

```bash
pip install pygame numpy
