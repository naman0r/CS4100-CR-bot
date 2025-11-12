# bot deck (CHOSEN DECK): knight, musketeer, bomber, archers, minions, giant, mini pekka, spear goblins

import random
import threading

import pyautogui
import time

NUM_BATTLES = 0
BATTLES_WON = 0
BATTLES_LOST = 0

'''
Returns true if the bot won, false otherwise
'''
def bot_won_battle():
    bot_win_location = pyautogui.locateCenterOnScreen(
        'win_state/bot_win.png', confidence=0.8, grayscale=True
    )
    return bot_win_location is not None
    
 
def winner_detected():
    global BATTLES_WON, BATTLES_LOST

    # Check if the OK button (end of match screen) is visible
    ok_location = pyautogui.locateCenterOnScreen('buttons/ok.png', confidence=0.8, grayscale=True)
    if ok_location is None:
        return False  # Match still ongoing

    print("OK button detected — match ended.")

    # Once the match is over, check if the win screen was shown
    bot_win_location = pyautogui.locateCenterOnScreen('win_state/bot_win_3.png', confidence=0.8, grayscale=True)
    if bot_win_location is not None:
        BATTLES_WON += 1
        print("Bot WON the battle!")
    else:
        BATTLES_LOST += 1
        print("Bot LOST the battle!")

    # Click OK to continue
    pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.1)
    pyautogui.click()

    return True

        
def play_card():
    card_slots = [(900, 900), (1000, 900), (1100, 900), (1200, 900)]
    my_arena_zone = {'x': (800, 1200), 'y': (480, 750)}  # potential drop zones

    time.sleep(random.uniform(3, 5))  # wait a bit for elixir

    card = random.choice(card_slots)
    x_spot = random.randint(my_arena_zone['x'][0], my_arena_zone['x'][1])
    y_spot = random.randint(my_arena_zone['y'][0], my_arena_zone['y'][1])

    pyautogui.moveTo(card[0], card[1], duration=0.2)
    pyautogui.dragTo(x_spot, y_spot, duration=0.3, button='left')

def play_unranked_match(max_duration=300):  # 5 minutes 
    start_time = time.time()
    while True:
        if winner_detected():
            print("Match over — Winner detected.")
            open_mystery_box() # opens box (if won) before new battle
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
    except pyautogui.ImageNotFoundException:
        print('Battle button not found')
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


