''' debug file for Clash Royale win detection - use this to isolate the problem/test 
if you're having win-detection/crown reward-related bugs with smart_clash_bot.py '''

# what's working: match win/loss detection OK
# BUG: crown detection, when opp wins 3 crowns and bot wins none bot prints: Bot LOST the battle! Crowns: 2, Reward: 3 -> should have -2 reward
# FIXED ABOVE: bot scans the specific region where YOUR crowns are, so it no longer confuses opponent's crowns won for the crowns it (the bot) won
# CAN: Correctly detect losses, differentiate between opp's won crowns and it's (bot) own won crowns, detect 2 and 3 crowns rewards correctly
# BUG: Consistently misidentifies 1 crown as 2 crowns -> FIXED, INCREASE CONFIDENCE
# MAJOR PROBLEM WAS FIXED BY SCREENSHOTTING THE ORIGINAL GAME CROWN IMAGES - PREVIOUSLY,
# I TOOK SCREENSHOTS OF SCREENSHOTS OF THE GAME - THIS MESSED UP THE DIMENSIONS, 
# PYAUTOGUI COULDN'T FIND THEM IF ITS LIFE DEPENDED ON IT.

import random
import pyautogui
import time
from local_placements import bot_crown_region 

BATTLES_WON = 0
BATTLES_LOST = 0

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

time.sleep(5)
print(winner_detected())
