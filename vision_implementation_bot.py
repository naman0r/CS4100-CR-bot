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
# BOT DECK (use these cards): knight, firecracker (evo), valkyrie, witch, 
							  bomber (evo), furnace, mini pekka 

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
import os

from collections import defaultdict
from local_placements import card_slots, king_tower, princess_towers, bridge, bot_crown_region, opp_crown_region

# -----------------------------------
# 1. Define Card Names & Action Space
# -----------------------------------

# You must have a screenshot for each of these in the cards/ folder!
# NOTE: We handle the "Evo" duplicates by mapping them to the same logical card name
# or treating them as separate strategy cards.
# Based on user's deck:
CARD_NAMES = [
    "knight", 
    "firecracker", 
    "valkyrie", # Matches filename
    "witch", 
    "bomber", 
    "furnace", 
    "minipekka", # Matches filename
    "skeletons"
]

# Map for handling Evo variants: Logic Name -> List of possible filenames
# Logic Name is what the RL learns. Filenames are what we look for.
CARD_VARIANTS = {
    "knight": ["knight.png"],
    "firecracker": ["firecracker.png", "firecracker_evo.png"],
    "valkyrie": ["valkarie.png"],
    "witch": ["witch.png"],
    "bomber": ["bomber.png", "bomber_evo.png"],
    "furnace": ["furnace.png"],
    "minipekka": ["minipekka.png"],
    "skeletons": ["skeletons.png"]
}

zones = {
    "king_tower": king_tower,
    "princess_towers": princess_towers,
    "bridge": bridge
}

# Action = (Card_Name, Zone, Side)
# e.g. ("tesla", "bridge", "left")
actions = []
for card in CARD_NAMES:
    for zone in zones:
        for side in zones[zone]:
            actions.append((card, zone, side))



Q = defaultdict(float)
N = defaultdict(int)
gamma = 0.9
epsilon = 0.9 
decay_rate = 0.995

NUM_BATTLES = 0
BATTLES_WON = 0
BATTLES_LOST = 0

last_action = None
last_state = None
battle_history = []
match_durations = []

# Load Q-table if exists
try:
    with open("vision_qtable.pkl", "rb") as f:
        Q.update(pickle.load(f))
        print(f"\n[INFO] Loaded Vision Q-table with {len(Q)} entries.")
except FileNotFoundError:
    print("[INFO] No saved Vision Q-table found, starting fresh.")


def safe_exit():
    print("\n[SAFE EXIT] Saving Q-table before quitting...", flush=True)
    save_Q() 
    print("[SAFE EXIT] Q-table saved. Exiting now.", flush=True)


# ---------------------------
# 2. Vision Helper Functions
# ---------------------------

def identify_card_in_slot(slot_idx):
    """
    Takes a screenshot of the card slot and compares it against reference images in cards/
    Returns the name of the card found (e.g. "firecracker"), or None if no match.
    """
    x, y = card_slots[slot_idx]
    # Define region slightly larger than the card to be safe
    # Slot center is x,y. 
    search_region = (x - 40, y - 50, 80, 100) 
    
    # Iterate through all LOGICAL cards in our deck
    for logic_name, variants in CARD_VARIANTS.items():
        # Check each variant (e.g. Normal, Evo)
        for filename in variants:
            try:
                img_path = f"cards/{filename}"
                if not os.path.exists(img_path): 
                    continue
                    
                # confidence=0.7 is a good starting point for small icons
                if pyautogui.locateOnScreen(img_path, region=search_region, confidence=0.7, grayscale=True):
                    return logic_name # Return the LOGICAL name (e.g. "firecracker")
            except Exception:
                pass
            
    return None

def get_hand_cards():
    """
    Returns a dict: { "firecracker": 0, "knight": 2 } mapping Card Name -> Slot Index.
    """
    hand = {}
    for i in range(4):
        card_name = identify_card_in_slot(i)
        if card_name:
            hand[card_name] = i
    return hand



# ---------------------------
# Q-learning helper functions
# ---------------------------
def get_state(start_time):
    """
    Define state based on elapsed time AND threat detection.
    Returns tuple: (game_phase, threat_side)
    """
    elapsed = time.time() - start_time
    if elapsed < 60:
        phase = "early"
    elif elapsed < 120:
        phase = "mid"
    else:
        phase = "late"
    
    return phase
    #threat = detect_threats()
    #return (phase, threat)

def choose_action(state, current_hand):
    """
    Epsilon-greedy selection of (Card_Name, Zone, Side).
    """
    # Filter actions to only those cards currently in our hand
    # AND (optionally) check if they are available (elixir check)
    
    available_actions = []
    for action in actions:
        card_name = action[0]
        if card_name in current_hand:
            slot_idx = current_hand[card_name]
            if is_card_available(slot_idx):
                available_actions.append(action)
    
    # Safety fallback
    if not available_actions:
        # Return a random action even if not possible (prevents crash)
        return random.choice(actions)

    # Epsilon-Greedy
    if random.random() < epsilon:
        return random.choice(available_actions)
    
    return max(available_actions, key=lambda a: Q[(state, a)])

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
    with open("vision_qtable.pkl", "wb") as f:
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


'''
# ---------------------------
# Computer Vision / Perception
# ---------------------------
def detect_threats():
    """
    Scans left and right lane regions for enemy troops (Red color).
    Returns: "none", "left", "right", or "both"
    """
    # Define regions based on princess_towers and bridge
    # Left Lane Box: Approx between Left Princess Tower and Bridge
    # These coords are estimates based on local_placements.py
    left_region = (800, 470, 160, 150) # x, y, w, h
    right_region = (1080, 470, 160, 150)
    
    threats = []
    
    for side, region in [("left", left_region), ("right", right_region)]:
        try:
            img = pyautogui.screenshot(region=region)
            arr = np.array(img)
            
            # Simple Red Detection
            # R > 150, G < 100, B < 100 (heuristic for red health bars/troops)
            red_mask = (arr[:,:,0] > 150) & (arr[:,:,1] < 100) & (arr[:,:,2] < 100)
            red_pixels = np.sum(red_mask)
            
            # If more than 50 red pixels, consider it a threat
            if red_pixels > 50:
                threats.append(side)
                
        except Exception as e:
            print(f"[WARNING] Threat detection failed: {e}", flush=True)

    if "left" in threats and "right" in threats:
        return "both"
    elif "left" in threats:
        return "left"
    elif "right"	 in threats:
        return "right"
    else:
        return "none"
'''

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

    ''' reward bot defensiveness (not losing all princess towers) '''
    if pyautogui.locateOnScreen('win_state/opp_three_crown.png',
                                confidence=0.90, # high confidence to ensure match
                                grayscale=True,
                                region=opp_crown_region):
	    print("Opponent three-crowned bot, no bot rewards", flush=True)

    elif pyautogui.locateOnScreen('win_state/opp_two_crown.png',
                                confidence=0.90, # high confidence to ensure match
                                grayscale=True,
                                region=opp_crown_region):
	    reward += 3
	    print("+3 reward, bot saved king tower", flush=True)
	    						
    elif pyautogui.locateOnScreen('win_state/opp_one_crown.png',
                                confidence=0.90, # high confidence to ensure match
                                grayscale=True,
                                region=opp_crown_region):
        reward += 5
        print("+5 reward, bot saved one princess tower and king tower", flush=True)
    
    else:
        reward += 8
        print("+8 reward, bot saved all towers", flush=True)
                

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
        reward += -6
        BATTLES_LOST += 1
        print(f"Bot LOST the battle! Crowns: {crowns}, Reward: {reward}", flush=True)
        log_battle_result(reward, False)

    pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.2)
    pyautogui.click()
    
    # Generate graphs
    plot_progress()
    #plot_match_durations()
    
    return reward   
     
# ---------------------------
# 4. Game Loop Functions
# ---------------------------

def play_card(start_time):
    """Play a card using RL + Vision to choose best Card + Zone + Side."""
    global last_action, last_state

    # Wait for elixir / cooldown
    # Slightly randomized to avoid detection
    time.sleep(random.uniform(2, 4)) 

    # 1. Identify Cards in Hand (Vision Step)
    current_hand = get_hand_cards()
    
    # If hand is empty (recognition failed or no cards match), wait and try again
    if not current_hand:
        # print("[VISION] No known cards identified in hand...", flush=True)
        return # Skip turn

    # 2. Determine current state
    current_state = get_state(start_time)

    # 3. Update Q-table for previous action if we are continuing the game
    if last_action is not None and last_state is not None:
        update_Q(last_state, last_action, 0, current_state)

    # 4. RL chooses best action from AVAILABLE cards in hand
    action = choose_action(current_state, current_hand)
    
    card_name, zone_name, side = action
    
    # 5. Execute Action
    # Map logical card name back to physical slot index
    if card_name not in current_hand:
        # Should not happen if choose_action works correctly
        return

    slot_idx = current_hand[card_name]
    
    last_action = action
    last_state = current_state
    
    # Physical Click
    card_pos = card_slots[slot_idx]
    target_pos = zones[zone_name][side]

    # Perform drag from card slot to placement
    pyautogui.moveTo(card_pos[0], card_pos[1], duration=0.2)
    pyautogui.dragTo(target_pos[0], target_pos[1], duration=0.5, button="left")

    print(f"[ACTION] State: {current_state} | Played {card_name} (Slot {slot_idx}) at {zone_name} ({side})", flush=True)


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

