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


# All possible (zone, side) combinations
zones = {
    "king_tower": king_tower,
    "princess_towers": princess_towers,
    "bridge": bridge
}

actions = [(zone, side) for zone in zones for side in zones[zone]]  # e.g. ("bridge", "left")
last_action = None

# Load Q-table if exists
try:
    with open("zone_qtable.pkl", "rb") as f:
        Q.update(pickle.load(f))
        print(f"[INFO] Loaded Q-table with {len(Q)} entries.")
except FileNotFoundError:
    print("[INFO] No saved Q-table found, starting fresh.")

# ---------------------------
# Q-learning helper functions
# ---------------------------
def choose_action():
    """greedy selection of (zone, side)"""
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[(state, a)])

def update_Q(action, reward):
    """Q-learning update"""
    N[(state, action)] += 1
    eta = 1 / (1 + N[(state, action)])
    best_next = max(Q[(state, a)] for a in actions)
    Q[(state, action)] += eta * (reward + gamma * best_next - Q[(state, action)])

def save_Q():
    """Persist learned Q-values"""
    with open("zone_qtable.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    """Persist battles won and lost"""
    with open("battle_stats.txt", "a") as f:
        f.write(f"{NUM_BATTLES},{BATTLES_WON},{BATTLES_LOST}\n")

'''
Returns true if the bot won, false otherwise
'''
def bot_won_battle():
    bot_win_location = pyautogui.locateCenterOnScreen(
        'win_state/bot_win_3.png', confidence=0.8, grayscale=True
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

    print("OK button detected - Match ended, letting confetti fall")
    time.sleep(9)  # Let confetti fall
	
	# NOTE: UPDATE CROWN_REGION (in local_placements.py) to work for your screen
	# dimensions! 

    if pyautogui.locateOnScreen('win_state/three_crown.png',
                                confidence=0.95, # high confidence to ensure match
                                grayscale=True,
                                region=bot_crown_region):
        crowns, reward = 3, 6
    
    elif pyautogui.locateOnScreen('win_state/two_crown.png',
                                  confidence=0.95,
                                  grayscale=True,
                                  region=bot_crown_region): 
        crowns, reward = 2, 3
    
    elif pyautogui.locateOnScreen('win_state/one_crown.png',
                                  confidence=0.95,
                                  grayscale=True,
                                  region=bot_crown_region):
        crowns, reward = 1, 1
    
    else:
        crowns, reward = 0, -2

	# You may need to change below code to win_state/bot_win_2.png or 3, 
	# (or create your own screenshot/put it here), depending on arena background
	# Check the win_state folder to change the bot's win image
    bot_win_location = pyautogui.locateCenterOnScreen(
        'win_state/bot_win_3.png', confidence=0.8, grayscale=True
    )
    if bot_win_location:
        reward += 8
        BATTLES_WON += 1
        print(f"Bot WON the battle! Crowns: {crowns}, Reward: {reward}")
    else:
        BATTLES_LOST += 1
        print(f"Bot LOST the battle! Crowns: {crowns}, Reward: {reward}")

    pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.2)
    pyautogui.click()
    return reward 
    
    
    '''Q_learning.py code below, for graphs/figures of learning. Andrew -> add the 
    necessary code to make below plots work 
        
	# --- Plot rewards per episode ---
	plt.figure(figsize=(10, 6))

	plt.plot(rewards_per_episode, color='lightgray', linewidth=1, label='Raw Reward')
	plt.plot(range(window_size - 1, len(running_avg) + window_size - 1),
			 running_avg, color='steelblue', linewidth=2, label=f'Running Avg ({window_size})')

	plt.title(f"Training Rewards per Episode\nEpisodes={num_episodes}, Decay={decay_rate}", fontsize=14)
	plt.xlabel("Episode", fontsize=12)
	plt.ylabel("Total Reward", fontsize=12)
	plt.legend(fontsize=10)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(f"rewards_plot_{num_episodes}_{decay_rate}.png", dpi=300)
	plt.close()    
    '''

def play_card():
    """Play a random card using RL to choose best zone + side."""
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

def play_unranked_match(max_duration=300):  # 5 minutes 
    start_time = time.time()
    while True:
        reward = winner_detected()
        if reward is not None:
            print("Match over — Winner detected.")
            update_Q(last_action, reward)
            save_Q()
            epsilon = max(0.05, epsilon * decay_rate) # prevent hitting 0
            open_mystery_box()
            break
        elif time.time() - start_time > max_duration:
            print("Match timeout.")
            break
        else:
            play_card()

def start_battle():
    print("Starting battle!")
    try:
        battle_location = pyautogui.locateCenterOnScreen('buttons/battle.png', confidence=0.8,
                                                         grayscale=True)
        pyautogui.moveTo(battle_location.x, battle_location.y, duration=0.1)
        pyautogui.click()
        return True
    except Exception as e:
        print(f"Battle button not found: {e}")
        return False

def open_mystery_box():
    time.sleep(2)
    print("Opening mystery box...")
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
                print(f'--- Played {battles} battles! ---')
                break
            except ValueError:
                print("Please enter a valid number.")
        elif mode == '2':
            while True:
                run_app()
        else:
            print("Invalid option. Try again.")

'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 5000000
decay_rate = 0.995

def run_app():
    global NUM_BATTLES
    if NUM_BATTLES == 0:
        time.sleep(5) # time for opening CR

    if start_battle():
        time.sleep(3)  # time for letting battle load
        NUM_BATTLES += 1
        print(f'Playing battle #{NUM_BATTLES}')
        if NUM_BATTLES >= 2:
            print(f'Won: {BATTLES_WON}, Lost: {BATTLES_LOST}')
        play_unranked_match()

choose_battle_option()


