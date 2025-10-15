from flask import Flask, jsonify, request
from flask_cors import CORS
import rlcard
from rlcard.agents import RandomAgent
import time

app = Flask(__name__)
CORS(app)

# Initialize RLCard environment
env = rlcard.make('blackjack')
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)

# Game state storage
game_states = {}

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

@app.route('/')
def index():
    """Serve the main HTML page"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Blackjack</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #0f4c3a; 
            color: white; 
            margin: 0; 
            padding: 20px; 
            text-align: center;
        }
        .game-container {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
        }
        h1 {
            color: #ffd700;
            margin-bottom: 20px;
        }
        .game-area {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }
        .player, .dealer {
            flex: 1;
            margin: 0 10px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
        }
        .score {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .cards {
            min-height: 80px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            justify-content: center;
        }
        .card {
            background: white;
            color: black;
            width: 50px;
            height: 70px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .card.red {
            color: #d32f2f;
        }
        .card.black {
            color: #333;
        }
        .buttons {
            margin: 20px 0;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            margin: 0 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .message {
            font-size: 1.2em;
            font-weight: bold;
            margin: 20px 0;
            min-height: 30px;
        }
        .message.win {
            color: #4CAF50;
        }
        .message.lose {
            color: #f44336;
        }
        .message.tie {
            color: #ff9800;
        }
    </style>
</head>
<body>
    <div class="game-container">
        <h1>🃏 Blackjack Game</h1>
        
        <div class="game-area">
            <div class="dealer">
                <div class="score">Dealer: <span id="dealer-score">0</span></div>
                <div class="cards" id="dealer-cards"></div>
            </div>
            
            <div class="player">
                <div class="score">Player: <span id="player-score">0</span></div>
                <div class="cards" id="player-cards"></div>
            </div>
        </div>
        
        <div class="message" id="message">Click "New Game" to start!</div>
        
        <div class="buttons">
            <button onclick="startNewGame()">New Game</button>
            <button onclick="hitCard()" id="hit-btn" disabled>Hit</button>
            <button onclick="standCard()" id="stand-btn" disabled>Stand</button>
        </div>
    </div>

    <script>
        var gameId = null;
        var gameOver = true;
        
        var cardSymbols = { 'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣' };
        
        function getCardDisplay(card) {
            if (card.length === 2) {
                var suit = card[0];
                var rank = card[1];
                var isRed = (suit === 'H' || suit === 'D');
                var colorClass = isRed ? 'red' : 'black';
                
                return '<div class="card ' + colorClass + '">' + rank + cardSymbols[suit] + '</div>';
            }
            return '<div class="card black">' + card + '</div>';
        }
        
        function updateDisplay(data) {
            console.log('Updating display:', data);
            
            var state = data.state;
            var dealerCards = state.raw_obs['dealer hand'] || [];
            var playerCards = state.raw_obs['player0 hand'] || [];
            var playerScore = state.obs[0];
            var dealerScore = state.obs[1];
            
            // Update scores
            document.getElementById('player-score').textContent = playerScore;
            document.getElementById('dealer-score').textContent = dealerScore;
            
            // Update cards
            var dealerCardsHtml = dealerCards.map(getCardDisplay).join('');
            var playerCardsHtml = playerCards.map(getCardDisplay).join('');
            
            document.getElementById('dealer-cards').innerHTML = dealerCardsHtml;
            document.getElementById('player-cards').innerHTML = playerCardsHtml;
            
            // Update message
            var message = data.message || '';
            var messageEl = document.getElementById('message');
            messageEl.textContent = message;
            
            if (message.includes('Win')) {
                messageEl.className = 'message win';
            } else if (message.includes('Lose')) {
                messageEl.className = 'message lose';
            } else if (message.includes('Tie')) {
                messageEl.className = 'message tie';
            } else {
                messageEl.className = 'message';
            }
            
            // Update button states
            gameOver = data.game_over || false;
            document.getElementById('hit-btn').disabled = gameOver;
            document.getElementById('stand-btn').disabled = gameOver;
        }
        
        function startNewGame() {
            console.log('Starting new game...');
            fetch('/api/new-game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('New game response:', data);
                gameId = data.game_id;
                gameOver = false;
                updateDisplay(data);
            })
            .catch(error => {
                console.error('Error starting new game:', error);
                alert('Error starting new game: ' + error.message);
            });
        }
        
        function hitCard() {
            if (gameOver) return;
            
            console.log('Hitting...');
            fetch('/api/hit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ game_id: gameId })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Hit response:', data);
                updateDisplay(data);
            })
            .catch(error => {
                console.error('Error hitting:', error);
                alert('Error hitting: ' + error.message);
            });
        }
        
        function standCard() {
            if (gameOver) return;
            
            console.log('Standing...');
            fetch('/api/stand', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ game_id: gameId })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Stand response:', data);
                updateDisplay(data);
            })
            .catch(error => {
                console.error('Error standing:', error);
                alert('Error standing: ' + error.message);
            });
        }
    </script>
</body>
</html>
    '''

@app.route('/api/new-game', methods=['POST'])
def new_game():
    """Start a new game"""
    try:
        # Reset environment
        env.reset()
        
        # Get initial state
        state = env.get_state(0)
        game_id = str(int(time.time() * 1000))
        
        # Store game state
        game_states[game_id] = {
            'env': env,
            'state': state
        }
        
        # Convert numpy types for JSON serialization
        state_clean = convert_numpy_types(state)
        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'state': state_clean,
            'message': 'New game started!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hit', methods=['POST'])
def hit():
    """Player hits"""
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        
        if game_id not in game_states:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 400
        
        # Get game state
        env = game_states[game_id]['env']
        
        # Player hits (action 0)
        env.step(0)
        
        # Get new state
        state = env.get_state(0)
        game_states[game_id]['state'] = state
        
        # Check if game is over
        game_over = env.is_over()
        
        # Convert numpy types
        state_clean = convert_numpy_types(state)
        
        message = ""
        if game_over:
            # Calculate final result
            player_score = state['obs'][0]
            dealer_score = state['obs'][1]
            
            if player_score > 21:
                message = "You busted! Dealer wins!"
            elif dealer_score > 21:
                message = "Dealer busted! You win!"
            elif player_score > dealer_score:
                message = "You win!"
            elif player_score < dealer_score:
                message = "Dealer wins!"
            else:
                message = "It's a tie!"
        
        return jsonify({
            'success': True,
            'state': state_clean,
            'game_over': game_over,
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stand', methods=['POST'])
def stand():
    """Player stands"""
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        
        if game_id not in game_states:
            return jsonify({
                'success': False,
                'error': 'Game not found'
            }), 400
        
        # Get game state
        env = game_states[game_id]['env']
        
        # Player stands (action 1)
        env.step(1)
        
        # Get new state
        state = env.get_state(0)
        game_states[game_id]['state'] = state
        
        # Game is over after stand
        game_over = True
        
        # Convert numpy types
        state_clean = convert_numpy_types(state)
        
        # Calculate final result
        player_score = state['obs'][0]
        dealer_score = state['obs'][1]
        
        if player_score > 21:
            message = "You busted! Dealer wins!"
        elif dealer_score > 21:
            message = "Dealer busted! You win!"
        elif player_score > dealer_score:
            message = "You win!"
        elif player_score < dealer_score:
            message = "Dealer wins!"
        else:
            message = "It's a tie!"
        
        return jsonify({
            'success': True,
            'state': state_clean,
            'game_over': game_over,
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Simple Blackjack Game...")
    print("📱 Open your browser to: http://127.0.0.1:3001")
    print("🛑 Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=3001, debug=True)