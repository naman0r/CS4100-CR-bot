# Clash Royale Reinforcement Learning Bot

Group 8 Final Project for CS4100 at Northeastern University (Fall 2025)

## Project Description

This project implements an AI agent capable of playing the mobile game Clash Royale. The agent uses Reinforcement Learning (specifically Q-Learning) to learn optimal card placement strategies over time. The bot interacts with the game running on an Android emulator (BlueStacks) using screen capture and mouse automation.

## Underlying Technology

1. Python: The primary programming language used for all logic.
2. Reinforcement Learning (Q-Learning): The core AI algorithm. The agent maintains a Q-table mapping states (game time, hand) and actions (card, location) to expected rewards.
3. Computer Vision (OpenCV & PyAutoGUI):
   - Used to detect the game state (Early, Mid, Late game).
   - Used to detect match outcomes (Wins, Losses, Crowns).
   - Used to identify cards in the player's hand.
4. PyAutoGUI: Used to simulate mouse clicks and drags to place cards on the battlefield.
5. Matplotlib: Used to generate real-time performance graphs (Win Rate and Average Reward over time).

## Files and Architecture

### 1. smart_clash_bot.py (Baseline Agent)

This file contains the initial version of the Reinforcement Learning agent.

Approach:

- State Space: The game state is defined primarily by the "Game Phase" (Early Game: 0-60s, Mid Game: 60-120s, Late Game: 120s+).
- Action Space: The agent chooses a "Card Slot" (0, 1, 2, or 3) and a "Zone" (Bridge, Princess Tower, King Tower).
- Learning: The agent learns which _Slot Index_ is best to play in which zone.

Limitations:

- The bot does not know _what_ card it is playing, only which _slot_ it is clicking.
- This means if the deck order changes, the learned strategy becomes invalid.
- It effectively learns an "average" strategy for whatever cards happen to cycle through Slot 1, Slot 2, etc.

### 2. vision_implementation_bot.py (Improved Vision Agent)

This file contains the advanced version of the agent with Card Recognition.

Improvements:

- Card Recognition: We implemented a computer vision system that takes screenshots of the card slots and matches them against a reference library of card images.
- Knowledge Transfer: The Action Space was changed from (Slot Index) to (Card Name). The agent now learns "Play Knight at Bridge" instead of "Play Slot 1 at Bridge".
- Dynamic Slot Mapping: Before playing a card, the bot scans the current hand to find which slot that specific card is currently in.
- Evo Card Handling: The system handles "Evolved" card variants (which look different) by mapping multiple image templates to the same logical card name.

## Limitations of Current Approach

1. Latency: Taking screenshots and performing template matching introduces a delay (0.5 - 1.0 seconds) before every move. This makes the bot slower than a human player.
2. No Opponent Awareness: The bot currently only looks at its own hand and the clock. It does not effectively track enemy troops or their placement, making it purely offensive/blind.
3. Fixed Coordinates: The click coordinates are hardcoded for a specific window resolution (BlueStacks). If the window moves or resizes, the bot fails.
4. Simple State Space: The definition of "State" is very simple (Time elapsed). A more complex state including enemy troop count or elixir advantage would likely improve performance.

## Future Improvement Areas

1. Deep Reinforcement Learning (DQN): Replace the Q-table with a Neural Network. This would allow the agent to take the raw screen pixels as input, enabling it to "see" the entire battlefield and react to enemy troop positions.
2. Optimized Vision: Replace `pyautogui.locateOnScreen` with faster OpenCV direct matrix operations to reduce latency.
3. State Enrichment: Add logic to detect "Incoming Threat Left" or "Incoming Threat Right" so the bot can learn defensive maneuvers.
4. Deck Generalization: Train the model on multiple decks so it learns general unit interactions (e.g. "Splash damage counters swarms") rather than specific card names.

## Breakdown of Work:

- Andrew:

  - Wrote initial code for the random_clash_bot.py file, which eventually progressed to the smart_clash_bot.py file with our initial approach.
  - Did a lot of the setup, was the primary person to train the model and have it run on his computer.
  - Slide deck contributions

- Naman:
  - created github repository
  - implemented a large chunk of the AI-related functionalities in the smart_clash_bot.py file (Q-learning, state and action implementation, choose_action function).
  - Implemented action masking strategy in the smart_clash_bot.py file.
  - implemented vision_implementation_bot.py file, refining action tuple, and having a more sensible RL learning algorithm.
  - Wrote slides 6,7 and 8 in the slide deck (Environment, state, actions, action masking, Q-learning function and explaination.)
