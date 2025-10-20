# RL Project - Blackjack using RLCard
In this repo, we implement two model free RL agents (Q-Learning and Monte-Carlo) capable of learning how to play Blackjack (or at least the simplified version provided by the RLCard library). We also provide an interface which allows a human player to play against/with these agents.
 
##
### Authors
- Aron Kesete
- Isac Gustafsson
- Parna Saeidpour

##
### How to Run (using Make):

#### 0. Set up a Python virtual environment or Conda environment (Recommended)

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run complete solution
```bash
make all
```
This will run the training for both agents and start the interface for playing with them once training is finished.

#### 2. (Alt.) Run training and Interface separately
First, train the models using:
```bash
make train
```
It is possible to play againts untrained models, but we recommend training them to make things interesting.

Second, start the interface using:
```bash
make play
```
##
### How to Run (directly with Python):

#### Steps 0 and 1 are the same as for How to Run (using Make)

#### 2. Train the Q-Learning agent
```bash
python analysis.py
```

#### 3. Train the Monte-Carlo agent
```bash
python mc.py
```

#### 4. Start the interface
```bash
python interface.py
```
