#algo: randomly select cards like last time, but improve their placements when deployed.
# if tesla or skeleton cards seen, place it in the center

import random
import pyautogui

def place_card():


    if skeleton or tesla:
    try:
        ok_location = pyautogui.locateCenterOnScreen('buttons/ok.png', confidence=0.8,
                                                     grayscale=True)
        pyautogui.moveTo(ok_location.x, ok_location.y, duration=0.1)
        pyautogui.click()
        pyautogui.dragTo(1000, 500, duration=0.5, button='left')
        return True
    except pyautogui.ImageNotFoundException:
        return False

    else:
        place_card_behind_king_tower()

def place_card_behind_king_tower():
    direction = ['left', 'right']
    rand_dir = random.choice(direction)

    if rand_dir == 'left':
        pyautogui.dragTo(990, 770, duration=0.3, button='left')
    elif rand_dir == 'right':
        pyautogui.dragTo(1020, 770, duration=0.3, button='left')
    else:
        print('bruh')
