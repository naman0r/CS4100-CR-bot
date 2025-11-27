# Readme for group 8's final project for CS4100 at Northeastern University (Fall 2025)

## Description of the project:

- AI Powered Clash Royale playing bot
- TODO: Add more about what AI algorithms we will use
- Reward/punishment explaination

### Functional requirements:

### Nice to haves:

### AI-concepts, description of environment:

- **Reinforcement Learning (Q-Learning)**: The agent learns an optimal policy through trial and error, updating its Q-table based on rewards.
- **State Space**:
  - **Temporal Phases**: Early Game (0-60s), Mid Game (60-120s), Late Game (120s+).
  - **Threat Detection**: The bot uses Computer Vision to detect enemy attacks (Red Health Bars) in the Left or Right lanes.
  - Combined State: `(Game_Phase, Threat_Direction)` (e.g., `('late', 'left')`).
- **Action Space**: `(Card_Slot, Zone, Side)` - allowing the bot to choose _which_ card to play and _where_.
- **Reward Function**:
  - **Win**: +8 | **Crowns**: +1/3/6
  - **Loss**: -2
  - **Invalid Move (Unavailable Card)**: -0.5 (Learns resource management).

### Technologies going to be used:

- **Python**: Core logic.
- **OpenCV & NumPy**: Computer Vision for state detection (enemies) and resource validation.
- **PyAutoGUI**: Screen interaction.
- **Matplotlib**: Real-time performance analytics (Win Rate/Reward curves).

### Distribution of Work:

Andrew:

- Added original Clash Royale bot skeleton code
- Improved above code (added input validation, match timeouts, etc.)
- Added win detection/win-loss tracking
