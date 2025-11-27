# Idea:
# Train bot to learn the optimal card-placement strategy from the below locations:
# - Behind the king tower (lowest placement)
# - In front of the princess towers (medium)
# - In front of the bridge (highest)
# For each of the above listed placements, the bot has a 50% chance of placing
# a card either to the left or right of the aforementioned location (e.g. 50/50
# left or right side behind the king tower, 50/50 left or right bridge, etc.)

# Over time, the bot will learn which zone and left/right combinations tend to 
# win more games.

'''
# BOT DECK (use these cards): knight, musketeer, bomber, archers, minions, 
							  giant, mini pekka, spear goblins

# Note: Let's model this clash bot after Programming Assignment 2 - RL (Q_learning.py)

Group TODO:
	- andrew: debug crown rewards, (done) -> add file i/o to track wins losses rewards
	- addison: play game to get familiar with mechanics, run program
	- philip: setup bluestacks clash royale and think of graphs/figures (attributes) for presentation
	- naman: CR gym enviornment and figure how to incorporate RL into project 
	- EVERYONE: Add coordinate placements for your computer in local_placements.py
'''

# Tracks total wins/losses of each match
# Tracks succesfully defeated princess towers

# reward is based on number of crowns won (0 to 3) 
# reward is like 1 crown +1 reward, 2 crowns +3 reward, 3 crowns +6 reward

# winning the battle should give bonus reward (+8 or something). Max reward 
# yielded when bot gets most crowns (3) AND wins the match

# remember: winning 2 crowns does not necessarily mean you won the match. You can
# defeat two princess towers but still lose


import random
import threading
import pyautogui
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from collections import defaultdict
from local_placements import card_slots, king_tower, princess_towers, bridge, bot_crown_region

Q = defaultdict(float)
N = defaultdict(int)
gamma = 0.9
epsilon = 0.3   # start with 30% exploration
#decay_rate variable below

NUM_BATTLES = 0
BATTLES_WON = 0
BATTLES_LOST = 0


# All possible (card_index, zone, side) combinations
# We have 4 card slots (0, 1, 2, 3)
card_indices = range(4)
zones = {
    "king_tower": king_tower,
    "princess_towers": princess_towers,
    "bridge": bridge
}

# State is now dynamic based on game time
# state = "battle" (Removed single state)

actions = [(card_idx, zone, side) for card_idx in card_indices for zone in zones for side in zones[zone]]
last_action = None
last_state = None

match_durations = []   # (episode, duration_seconds)


# Load Q-table if exists
try:
    with open("zone_qtable.pkl", "rb") as f:
        Q.update(pickle.load(f))
        print(f"\n[INFO] Loaded Q-table with {len(Q)} entries.")
except FileNotFoundError:
    print("[INFO] No saved Q-table found, starting fresh.")


def safe_exit():
    print("\n[SAFE EXIT] Saving Q-table before quitting...", flush=True)
    save_Q() 
    print("[SAFE EXIT] Q-table saved. Exiting now.", flush=True)


# ---------------------------
# Q-learning helper functions
# ---------------------------
def get_state(start_time):
    """
    Define state based on elapsed time.
    0: Early Game (0-60s)
    1: Mid Game (60-120s)
    2: Double Elixir / Late Game (120s+)
    """
    elapsed = time.time() - start_time
    if elapsed < 60:
        return "early"
    elif elapsed < 120:
        return "mid"
    else:
        return "late"

def choose_action(state):
    """
    Epsilon-greedy selection of (card_index, zone, side) with Action Masking.
    Only considers valid cards to speed up learning.
    """
    # Identify valid cards first (Action Masking)
    valid_actions = []
    for action in actions:
        card_idx = action[0]
        # We can't check vision for EVERY action in the loop (too slow),
        # so we'll do a standard epsilon check first, BUT
        # for the max() step, we should ideally only pick valid ones.
        # Optimization: Just check availability of the chosen card later?
        # NO, strict Action Masking means we filter the list of actions.
        # Let's just filter by card_idx validity ONCE per step if possible.
        pass 
        
    # Simplified: Use epsilon greedy on ALL actions, but if specific card is
    # invalid, the main loop applies penalty. 
    # Full masking is expensive here without caching vision results.
    
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[(state, a)])


def update_Q(current_state, action, reward, next_state):
    """
    Q-learning update based on class notes:
    Q(s, a) = (1 - eta) * Q(s, a) + eta * [r + gamma * V(s')]
    where V(s') = max_a' Q(s', a')
    """
    N[(current_state, action)] += 1
    eta = 1 / (1 + N[(current_state, action)])
    
    if next_state is None:
        # Terminal state, V(s') = 0
        best_next = 0
    else:
        best_next = max(Q[(next_state, a)] for a in actions)
        
    Q[(current_state, action)] = (1 - eta) * Q[(current_state, action)] + \
                                 eta * (reward + gamma * best_next)

def save_Q():
    """Persist learned Q-values"""
    with open("zone_qtable.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    """Persist battles won and lost"""
    with open("battle_stats.txt", "a") as f:
        f.write(f"{NUM_BATTLES},{BATTLES_WON},{BATTLES_LOST}\n")

# ---------------------------
# Visualization & Logging
# ---------------------------
battle_history = [] # List of (episode, won, reward)

def log_battle_result(reward, won):
    battle_history.append((NUM_BATTLES, 1 if won else 0, reward))
    # Save to CSV
    with open("battle_log.csv", "a") as f:
        f.write(f"{NUM_BATTLES},{1 if won else 0},{reward}\n")


def plot_progress():
    """Generates a graph of training progress"""
    if len(battle_history) < 2:
        return
        
    episodes = [x[0] for x in battle_history]
    wins = [x[1] for x in battle_history]
    rewards = [x[2] for x in battle_history]
    
    # Calculate moving average win rate (last 10 battles)
    window = 10
    win_rate = []
    for i in range(len(wins)):
        start = max(0, i - window + 1)
        win_rate.append(sum(wins[start:i+1]) / (i - start + 1))

    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, alpha=0.3, color='gray', label='Reward')
    plt.plot(episodes, [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))], 
             color='blue', label=f'Avg Reward ({window})')
    plt.title("Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # Subplot 2: Win Rate
    plt.subplot(1, 2, 2)
    plt.plot(episodes, win_rate, color='green', label='Win Rate')
    plt.title(f"Win Rate (Moving Avg {window})")
    plt.xlabel("Episode")
    plt.ylim(0, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()
    print("[INFO] Updated training_progress.png", flush=True)

# ---------------------------
# Computer Vision / Perception
# ---------------------------
def is_card_available(card_idx):
    """
    Checks if a card is ready to play (colorful) or on cooldown/no elixir (grayscale).
    Returns True if available, False otherwise.
    """
    x, y = card_slots[card_idx]
    # Capture a small region around the card center (10x10 pixels)
    region = (x - 5, y - 5, 10, 10)
    try:
        img = pyautogui.screenshot(region=region)
        # Convert to numpy array
        arr = np.array(img)
        # Calculate standard deviation of color channels
        # High std dev = colorful (R!=G!=B). Low std dev = grayscale (R~=G~=B)
        std_dev = np.std(arr, axis=2).mean()
        
        # Threshold: Grayscale usually has very low std dev (< 5-10)
        # Colorful cards usually have high std dev (> 20)
        is_ready = std_dev > 15 
        return is_ready
    except Exception as e:
        print(f"[WARNING] Vision check failed: {e}", flush=True)
        return True # Assume ready if check fails to avoid getting stuck

'''
Returns true if the bot won, false otherwise
'''
def bot_won_battle():
    bot_win_location = pyautogui.locateCenterOnScreen(
        'win_state/bot_win.png', confidence=0.8, grayscale=True
    )
    return bot_win_location is not None
    
 
def winner_detected():
    """
    Detects if the match has ended, updates wins/losses,
    counts crowns, and calculates reward.
    """
    global BATTLES_WON, BATTLES_LOST

    ok_location = pyautogui.locateCenterOnScreen(
        'buttons/ok.png', confidence=0.8, grayscale=True
    )
    if ok_location is None:
        return None  # Match still ongoing

    print("OK button detected - Match ended, letting confetti fall", flush=True)
    time.sleep(9)  # Let confetti fall
	
	# NOTE: UPDATE CROWN_REGION (in local_placements.py) to work for your screen
	# dimensions! 

    if pyautogui.locateOnScreen('win_state/three_crown.png',
                                confidence=0.90, # high confidence to ensure match
                                grayscale=True,
                                region=bot_crown_region):
        crowns, reward = 3, 8
    
    elif pyautogui.locateOnScreen('win_state/two_crown.png',
                                  confidence=0.90,
                                  grayscale=True,
                                  region=bot_crown_region): 
        crowns, reward = 2, 6
    
    elif pyautogui.locateOnScreen('win_state/one_crown.png',
                                  confidence=0.90,
                                  grayscale=True,
                                  region=bot_crown_region):
        crowns, reward = 1, 2
    
    else:
        crowns, reward = 0, -2

	# You may need to change below code to win_state/bot_win_2.png or 3, 
	# (or create your own screenshot/put it here), depending on arena background
	# Check the win_state folder to change the bot's win image
    bot_win_location = pyautogui.locateCenterOnScreen(
        'win_state/bot_win_3.png', confidence=0.8, grayscale=True
    )
    if bot_win_location:
        reward += 12
        BATTLES_WON += 1
        print(f"Bot WON the battle! Crowns: {crowns}, Reward: {reward}", flush=True)
        log_battle_result(reward, True)
    else:
        BATTLES_LOST += 1
        print(f"Bot LOST the battle! Crowns: {crowns}, Reward: {reward}", flush=True)
        log_battle_result(reward, False)

    pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.2)
    pyautogui.click()
    
    # Generate graphs
    plot_progress()
    #plot_match_durations()
    
    return reward   
     
def play_card(start_time):
    """Play a card using RL to choose best card_slot + zone + side."""
    global last_action, last_state

    # Wait for elixir / cooldown
    time.sleep(random.uniform(3, 5))

    # Determine current state
    current_state = get_state(start_time)

    # Update Q-table for previous action if we are continuing the game
    if last_action is not None and last_state is not None:
        # Small penalty for time passing or neutral reward? 
        # For now, 0 reward for surviving.
        update_Q(last_state, last_action, 0, current_state)

    # RL chooses best card + zone + side
    action = choose_action(current_state)
    
    card_idx, zone_name, side = action
    
    # PERCEPTION CHECK: Is the card actually ready?
    if not is_card_available(card_idx):
        # Punishment for trying to play unavailable card
        # Teach agent to wait or pick another card
        print(f"[VISION] Card {card_idx} unavailable. Penalty applied.", flush=True)
        update_Q(current_state, action, -0.5, current_state) # Negative reward
        return # Skip this turn, don't click
        
    last_action = action
    last_state = current_state
    
    # Select the card coordinates
    card = card_slots[card_idx]

    # Get coordinates from local_placements
    x_spot, y_spot = zones[zone_name][side]

    # Perform drag from card slot to placement
    pyautogui.moveTo(card[0], card[1], duration=0.3)
    pyautogui.dragTo(x_spot, y_spot, duration=0.5, button="left")

    print(f"[ACTION] State: {current_state} | Played Card {card_idx} at {zone_name} ({side})", flush=True)


'''
DO NOT DELETE - KEEP FOR FUTURE REFERENCE 
def play_card():
    # Play a random card using RL to choose best zone + side.
    global last_action

    # Wait for elixir / cooldown
    time.sleep(random.uniform(3, 5))

    # Select a random card from hand
    card = random.choice(card_slots)

    # RL chooses best zone + side
    action = choose_action()
    last_action = action
    zone_name, side = action

    # Get coordinates from local_placements
    x_spot, y_spot = zones[zone_name][side]

    # Perform drag from card slot to placement
    pyautogui.moveTo(card[0], card[1], duration=0.2)
    pyautogui.dragTo(x_spot, y_spot, duration=0.3, button="left")

    print(f"[ACTION] Played at {zone_name} ({side}) -> ({x_spot}, {y_spot})")
'''

'''
DO NOT DELETE - KEEP FOR FUTURE REFERENCE 
def play_card():
    time.sleep(random.uniform(3, 5))  # wait a bit for elixir

    # Pick a card from hand
    card = random.choice(card_slots)

    # Define possible zones
    zones = [king_tower, princess_towers, bridge]

    # Pick a random zone
    selected_zone = random.choice(zones)

    # Pick left or right within the selected zone
    x_spot, y_spot = random.choice(list(selected_zone.values()))

    # Move and drop the card
    pyautogui.moveTo(card[0], card[1], duration=0.2)
    pyautogui.dragTo(x_spot, y_spot, duration=0.3, button='left')
'''

def plot_match_durations():
    if len(match_durations) < 2:
        return

    episodes = [x[0] for x in match_durations]
    durations = [x[1] for x in match_durations]

    window = 10
    moving_avg = [
        np.mean(durations[max(0, i-window+1):i+1])
        for i in range(len(durations))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, durations, alpha=0.3, label="Match Duration (seconds)")
    plt.plot(episodes, moving_avg, linewidth=2, label=f"Moving Avg ({window})")
    plt.title("Match Duration Over Training Time")
    plt.xlabel("Episode")
    plt.ylabel("Duration (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("match_durations.png")
    plt.close()

    print("[INFO] Updated match_durations.png", flush=True)


def play_unranked_match(max_duration=300):  # 5 minutes 
    global epsilon, NUM_BATTLES

    start_time = time.time()
    while True:
        reward = winner_detected()
        if reward is not None:
            duration = time.time() - start_time

            # Record match duration
            match_durations.append((NUM_BATTLES, duration))
            with open("match_durations.csv", "a") as f:
                f.write(f"{NUM_BATTLES},{duration}\n")
		
            print(f"Match lasted {duration:.1f} seconds.", flush=True)
            plot_match_durations()
			
            # Q-learning update
            if last_action is not None and last_state is not None:
                update_Q(last_state, last_action, reward, None)

            save_Q()
            epsilon = max(0.05, epsilon * decay_rate)

            open_mystery_box()
            break

        elif time.time() - start_time > max_duration:
            duration = max_duration

            match_durations.append((NUM_BATTLES, duration))
            with open("match_durations.csv", "a") as f:
                f.write(f"{NUM_BATTLES},{duration}\n")

            print("Match timeout.", flush=True)
            break

        else:
            play_card(start_time)


def start_battle():
    print("\nStarting battle!", flush=True)
    try:
        battle_location = pyautogui.locateCenterOnScreen('buttons/battle.png', confidence=0.8,
                                                         grayscale=True)
        pyautogui.moveTo(battle_location.x, battle_location.y, duration=0.1)
        pyautogui.click()
        return True
    except Exception as e:
        print(f"Battle button not found: {e}", flush=True)
        return False

def open_mystery_box():
    time.sleep(2)
    print("Opening mystery box...", flush=True)
    pyautogui.moveTo(1000, 500)
    pyautogui.click(clicks=5, interval=0.7) 
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.click()

def choose_battle_option():
    print("NOTE: Ensure you are playing Classic 1v1 mode with this bot")
    while True:
        mode = input("\nPress 1 to play 'n' battles, press 2 for unlimited: ").strip()
        if mode == '1':
            try:
                battles = int(input('Enter number of battles: '))
                for _ in range(battles):
                    run_app()
                print(f'--- Played {battles} battles! ---', flush=True)
                break
            except ValueError:
                print("Please enter a valid number.")
        elif mode == '2':
            while True:
                run_app()
        else:
            print("Invalid option. Try again.")

# Specify number of episodes and decay rate for training and evaluation.

num_episodes = 5000000
decay_rate = 0.995

def run_app():
    global NUM_BATTLES
    try:
        if NUM_BATTLES == 0:
            time.sleep(5)  # time for opening CR
        
        if start_battle():
            time.sleep(4)  # let battle load
            NUM_BATTLES += 1
            print(f'Playing battle #{NUM_BATTLES}', flush=True)

            if NUM_BATTLES >= 2:
                print(f'Won: {BATTLES_WON}, Lost: {BATTLES_LOST}', flush=True)

            play_unranked_match()

    except KeyboardInterrupt:
        safe_exit()
        # Rethrow to stop outer loop if needed
        raise  

if __name__ == "__main__":
    try:
        choose_battle_option()
    except KeyboardInterrupt:
        safe_exit()
        print("Exited safely.")

