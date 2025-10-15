# RL_project
RL Project



### 1. Create Virtual Environment (Recommended)
```bash
python -m venv blackjack_env
source blackjack_env/bin/activate  # On Windows: blackjack_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Choose Your Interface!

####  **Terminal Game:**
```bash
python simple_game.py
```

####  **Web Game:**
```bash
python web_blackjack.py
```
Then open: **http://127.0.0.1:3001**

## How to Play

### Terminal Version:
- Run `python simple_game.py`
- Follow the prompts to choose Hit (0) or Stand (1)
- Play as many games as you want

### Web Version:
- Run `python web_blackjack.py`
- Open http://127.0.0.1:3001 in your browser
- Click "New Game" to start
- Use "Hit" and "Stand" buttons to play
- Clean and simple interface!

## Game Rules

- **Objective**: Get as close to 21 as possible without going over
- **Hit**: Take another card
- **Stand**: Keep your current hand
- **Dealer**: Must hit until 17 or higher
- **Winning**: Beat the dealer's score without busting