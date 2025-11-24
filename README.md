# AI Clash Royale Bot - Final Project

This project implements a Reinforcement Learning (Q-Learning) agent to play Clash Royale automatically.

## 🧠 AI Architecture

### 1. State Representation
The bot perceives the game environment through a temporal state system:
- **Early Game (0-60s)**: Conservative playstyle
- **Mid Game (60-120s)**: Normal engagement
- **Late Game (120s+)**: Double elixir / Aggressive play

### 2. Action Space
The agent has a discrete action space of `(Card_Slot, Zone, Side)`:
- **Card Slot**: 0-3 (Choosing *which* card to play)
- **Zone**: King Tower, Princess Tower, Bridge
- **Side**: Left, Right

### 3. Visual Perception & Validity
Unlike basic bots that click blindly, this agent uses **Computer Vision (OpenCV/Numpy)** to:
- Detect match end and crown results
- **Verify Elixir/Card Availability**: The bot checks color saturation of card slots.
    - If it attempts to play a gray (unavailable) card, it receives a **negative reward** (-0.5).
    - This forces the agent to learn resource management and valid moves.

### 4. Reward Function
- **Match Win**: +8
- **Crowns**: +1, +3, or +6 depending on count
- **Loss**: -2 (plus 0 crowns)
- **Invalid Move**: -0.5 (Trying to play unavailable card)

## 📊 Analytics

The bot automatically generates training graphs:
- **training_progress.png**: Visualizes Win Rate (Moving Average) and Reward History.
- **battle_log.csv**: Detailed log of every match result.

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the bot:
   ```bash
   python smart_clash_bot.py
   ```

3. Select "1" for a fixed number of battles or "2" for continuous training.

## 📂 File Structure
- `smart_clash_bot.py`: Main RL agent and game loop.
- `local_placements.py`: Screen coordinates (Config).
- `zone_qtable.pkl`: Saved Q-learning model.
