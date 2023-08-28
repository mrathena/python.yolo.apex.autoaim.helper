import time

from press_key import InputKey
from key_input import Mouse, Keyboard

#   用于测试这个包中press_key.py的每一个功能
if __name__ == '__main__':
    time.sleep(4)
    test_key = [Keyboard.W, Keyboard.A, Keyboard.S, Keyboard.D]

    for mode in range(2,3):
        print('mode:', mode)
        input_key = InputKey(mode)
        for key in test_key:
            print('press key:', key)
            input_key.press_key(key)
            time.sleep(0.5)
            input_key.release_key(key)
            time.sleep(1)
            input_key.click_key(key, 1)
            time.sleep(5)
        input_key.click_key(Keyboard.G, 0.1)
        time.sleep(3)
        print('press mouse key')
        input_key.mouse_move(300, 300)
        time.sleep(3)
        input_key.mouse_move(-100, -100)
        input_key.mouse_key_click(Mouse.MOUSE_LEFT)
        time.sleep(0.4)
        input_key.mouse_key_click(Mouse.MOUSE_RIGHT)
        time.sleep(0.4)
        input_key.mouse_key_click(Mouse.MOUSE_MIDDLE)
        time.sleep(0.4)
        input_key.click_key(Keyboard.LSHIFT, 0.02)
        input_key.mouse_scroll(1)
        time.sleep(0.5)
        input_key.mouse_scroll(-1)
        time.sleep(0.5)
        input_key.mouse_scroll(10)
        time.sleep(0.5)
        input_key.mouse_scroll(-10)
        input_key.click_key(Keyboard.LSHIFT, 0.1)
        time.sleep(1)
