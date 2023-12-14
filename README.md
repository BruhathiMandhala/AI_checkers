# AI_checkers
Final Project for AI

The project is to create a checkers game environment between Human vs AI and AI vs AI. (alpha beta pruning and Q learning Agent)
### About Checkers:
Checkers game is a traditional English Draughts game, which simply consists of 12 pieces for each player and can move according to the games.(red and blue for players)
To make a checkers gameplay using AI is the interesting task.
In this project, Human game play is defined and Alpha Beta Pruning game play is defined, and environment is set, so that there can be a interesting game between them.
This project also has AI vs AI gameplay between(Q-Learning Agent and Alpha Beta Pruning)

Alpha Beta Pruning: It is a an algorithm, that is used in game trees to read the no.of nodes evaluated. 
This keeps in track of alpha(best value of maximum player) and beta (best value for min player)

Q-Learning: It is a model-free reinforcement learning algorithm, that keeps on learning and updating.
It keeps the track of win rewards, lose rewars



## Checkers Game using AI
Two types of agents:
1. Alpha Beta Agent
2. Q Learning

Two modes of Playing:
1. Human vs AI
2. AI vs AI

### 1. Human vs AI
1. clone the repository
2. in the main branch run python HumanVsAI.py
3. First player is Alpha Beta agent and the second player is Human

### 2. AI vs AI
1. In the main branch run python AIvsAI.py
2. First player is Alpha-Beta and second is Q learning
3. can replace number of training rounds for Qlearning

### Conclusion:
To conclude this the Q-Learning agent has played better when it is put on a training loop.
This project can be further modelized to compare the Q-learning agent with other model-free Reinforcement agents
Analysing the different combination of moves and their weighted values are more important inunderstanding the game of checkers.
While designing the interface of game itself is not so difficult, designing an efficient way to build required more analysis.

### updated:
If the pieces are not visible as red and blue dots, change the players symbol in Board.py to o and x for normal pieces and O and X for kings
