import sys
import pyautogui
import time

def print_cursor_loc():
    print("Move your mouse. Press Ctrl+C to quit.\n")

    try:
        while True:
            x, y = pyautogui.position()
            sys.stdout.write(f"\rX: {x}  Y: {y}     ")
            sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")

print_cursor_loc()