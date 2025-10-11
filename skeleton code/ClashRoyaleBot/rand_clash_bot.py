import random
import threading

import pyautogui
import time

NUM_BATTLES = 0

def winner_detected():
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

def play_unranked_match():
    while True:
        if winner_detected():
            print("Match over — Winner detected.")
            open_mystery_box()
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
    mode = input('Press 1 to play \'n\' battles, press 2 to play unlimited battles: ')

    if mode == '1':
        battles = input('Enter the number of battles to play: ')
        for i in range(0, int(battles)):
            run_app()
        print(f'--- Played {battles} battles! ---')
    else:
        while True:
            run_app()

def run_app():
    global NUM_BATTLES
    if NUM_BATTLES == 0:
        time.sleep(3) # time for opening CR

    if start_battle():
        time.sleep(3)  # time for letting battle load
        NUM_BATTLES += 1
        print(f'Playing battle #{NUM_BATTLES}')
        play_unranked_match()

choose_battle_option()

# plays unranked clash mode, purpose: farm gold and see how good i can make it play
# double elixir starts after 3 min., set a timer for it to play cards faster during that time so
# it doesn't leak elixir
# use a bot-friendly deck (defensive cards, long-range so placement matters less)
# bot deck 1: mega knight, firecracker, bats, dart goblin, princess, furnace, skeletons, ice wizard
# bot deck 2: (evo)firecracker, (evo)tesla, ice wizard, witch, dart goblin, princess, furnace, skeletons
