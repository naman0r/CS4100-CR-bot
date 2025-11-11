
#calculates rewards for the reinforcement learning agent based on game state transitions and actions taken.


class RewardCalculator:
    def __init__(self):
        # Reward weights (tunable hyperparameters)
        self.TOWER_DAMAGE_WEIGHT = 1.0
        self.ENEMY_TOWER_DAMAGE_REWARD = 10.0  # per percentage point
        self.OWN_TOWER_DAMAGE_PENALTY = -8.0   # per percentage point
        
        self.TROOP_KILL_REWARD = 5.0
        self.TROOP_LOSS_PENALTY = -3.0
        
        self.ELIXIR_EFFICIENCY_WEIGHT = 0.5
        self.ELIXIR_WASTE_PENALTY = -2.0
        
        # Terminal rewards
        self.WIN_REWARD = 100.0
        self.LOSS_PENALTY = -100.0
        self.DRAW_REWARD = 0.0
        
        # Time-based penalties to encourage aggression
        self.TIMEOUT_PENALTY = -50.0
        self.STALLING_PENALTY = -0.1  # per second of inactivity
        
    def calculate_reward(self, prev_state, action, new_state):
        """
        Calculate reward for transitioning from prev_state to new_state after taking action.
        
        Args:
            prev_state: Dictionary containing previous game state
                - own_tower_health: float (0-100)
                - enemy_tower_health: float (0-100)
                - elixir: int (0-10)
                - troops_alive: list of troop objects
            action: Dictionary containing the action taken
                - card_placed: str (card name)
                - position: tuple (x, y)
                - elixir_cost: int
            new_state: Dictionary containing new game state (same structure as prev_state)
            
        Returns:
            float: Total reward for this transition
        """
        reward = 0.0
        
        # Tower damage rewards
        reward += self._calculate_tower_damage_reward(prev_state, new_state)
        
        # Troop trade rewards
        reward += self._calculate_troop_reward(prev_state, new_state)
        
        # Elixir efficiency rewards
        reward += self._calculate_elixir_reward(prev_state, action, new_state)
        
        # Check for terminal state (match end)
        if new_state.get('match_ended', False):
            reward += self._calculate_terminal_reward(new_state)
        
        return reward
    
    def _calculate_tower_damage_reward(self, prev_state, new_state):
        """Calculate reward based on tower health changes."""
        reward = 0.0
        
        # Reward for damaging enemy towers
        enemy_damage = prev_state['enemy_tower_health'] - new_state['enemy_tower_health']
        reward += enemy_damage * self.ENEMY_TOWER_DAMAGE_REWARD
        
        # Penalty for taking tower damage
        own_damage = prev_state['own_tower_health'] - new_state['own_tower_health']
        reward += own_damage * self.OWN_TOWER_DAMAGE_PENALTY
        
        return reward
    
    def _calculate_troop_reward(self, prev_state, new_state):
        """Calculate reward based on troop trades."""
        # TODO: Implement troop counting and comparison
        # - Count enemy troops destroyed
        # - Count own troops lost
        # - Calculate net value of trades
        return 0.0
    
    def _calculate_elixir_reward(self, prev_state, action, new_state):
        """Calculate reward based on elixir efficiency."""
        reward = 0.0
        
        # TODO: Implement elixir efficiency calculation
        # - Penalize wasting elixir (placing card that does nothing)
        # - Reward positive elixir trades
        # - Consider elixir advantage over time
        
        return reward
    
    def _calculate_terminal_reward(self, final_state):
        """Calculate reward for match outcome."""
        if final_state.get('won', False):
            return self.WIN_REWARD
        elif final_state.get('lost', False):
            return self.LOSS_PENALTY
        else:
            return self.DRAW_REWARD


# Configuration for easy tuning during training
REWARD_CONFIG = {
    'tower_damage_weight': 1.0,
    'enemy_tower_damage_reward': 10.0,
    'own_tower_damage_penalty': -8.0,
    'troop_kill_reward': 5.0,
    'troop_loss_penalty': -3.0,
    'elixir_efficiency_weight': 0.5,
    'elixir_waste_penalty': -2.0,
    'win_reward': 100.0,
    'loss_penalty': -100.0,
    'draw_reward': 0.0,
    'timeout_penalty': -50.0,
    'stalling_penalty': -0.1
}