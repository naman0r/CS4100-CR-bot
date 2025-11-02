import random
import threading

import pyautogui
import time

NUM_BATTLES = 0
BATTLES_WON = 0


def bot_won_battle():   
    try:
        bot_win_location = pyautogui.locateCenterOnScreen('win_state/bot_win.png', 
													confidence=0.8, grayscale=True)
        return True
    except pyautogui.ImageNotFoundException:
        return False
 
 
def winner_detected():
    global BATTLES_WON
    
    if bot_won_battle():
        BATTLES_WON += 1
			
    try:
        ok_location = pyautogui.locateCenterOnScreen('buttons/ok.png', confidence=0.8,
													grayscale=True)
        pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.1)
        pyautogui.click()
        return True
    except pyautogui.ImageNotFoundException:
        return False

def play_card():
    card_slots = [(900, 900), (1000, 900), (1100, 900), (1200, 900)]
    my_arena_zone = {'x': (800, 1200), 'y': (480, 750)}  # potential drop zones

    #timer = threading.Timer(180, say_hello) #double elixir place cards faster
    #timer.start()

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
        battle_location = pyautogui.locateCenterOnScreen('buttons/battle_bonus.png', confidence=0.8,
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
    while True:
        mode = input("Press 1 to play 'n' battles, press 2 for unlimited: ").strip()
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
        time.sleep(3) # time for opening CR

    if start_battle():
        time.sleep(3)  # time for letting battle load
        NUM_BATTLES += 1
        print(f'Playing battle #{NUM_BATTLES}')
        if NUM_BATTLES >= 2:
            print(f'Won: {BATTLES_WON}, Lost: {NUM_BATTLES - BATTLES_WON}')
        play_unranked_match()

choose_battle_option()

# plays unranked clash mode, purpose: farm gold and see how good i can make it play
# double elixir starts after 3 min., set a timer for it to play cards faster during that time so
# it doesn't leak elixir
# use a bot-friendly deck (defensive cards, long-range so placement matters less)
# bot deck 1: mega knight, firecracker, bats, dart goblin, princess, furnace, skeletons, ice wizard
# bot deck 2: (evo)firecracker, (evo)tesla, ice wizard, witch, dart goblin, princess, furnace, skeletons
