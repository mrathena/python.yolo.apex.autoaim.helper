import ctypes
import os
import time
from key_input import Keyboard, Key, MouseEvent, Mouse, MouseKey
import pyautogui

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


class InputKey:
    WHEEL_DELTA = 120

    def __init__(self, mode: int = 0):
        self.driver_dd = None
        self.mode = mode
        self.change_mode(mode)

    def change_mode(self, mode):
        self.mode = mode
        if mode < 0 or mode > 2:
            self.mode = 0
        if mode == 1:
            try:
                root = os.path.abspath(os.path.dirname(__file__))
                self.driver_logi = ctypes.CDLL(f'{root}/logitech.driver.dll')
                ok = self.driver_logi.device_open() == 1
                if not ok:
                    print('Error, GHUB or LGS driver not found')
                    self.mode = 0
                else:
                    self.mode = 1
            except FileNotFoundError:
                print(f'Error, DLL file not found.Then use default mode.')
                self.mode = 0
        if mode == 2:
            try:
                root = os.path.abspath(os.path.dirname(__file__))

                self.driver_dd = ctypes.windll.LoadLibrary(f'{root}/dd_driver.dll')
                time.sleep(2)
                st = self.driver_dd.DD_btn(0)
                ok = st == 1
                if not ok:
                    print('Error, DD94687.64.dll not found')
                    self.mode = 0
                else:
                    self.mode = 2
            except FileNotFoundError:
                print(f'Error, DLL file not found.Then use default mode.')
                self.mode = 0

    def press_key(self, key: Key):
        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, key.scan_code, 0x0008, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.key_down(key.char.lower())
        elif self.mode == 2:
            self.driver_dd.DD_key(key.dd_code, 1)

    def release_key(self, key: Key):
        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, key.scan_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.key_up(key.char.lower())
        elif self.mode == 2:
            self.driver_dd.DD_key(key.dd_code, 2)

    def click_key(self, key: Key, delay: float = 0.02):
        self.press_key(key)
        time.sleep(delay)
        self.release_key(key)

    def mouse_key_press(self, key: MouseKey):
        if key == Mouse.MOUSE_MOVE or key == Mouse.MOUSE_WHEEL:
            return

        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, key.press_virtual_code, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.mouse_down(key.press_logi_code)
        elif self.mode == 2:
            self.driver_dd.DD_btn(key.press_dd_code)

    def mouse_key_release(self, key: MouseKey):
        if key == Mouse.MOUSE_MOVE or key == Mouse.MOUSE_WHEEL:
            return

        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, key.release_virtual_code, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.mouse_up(key.release_logi_code)
        elif self.mode == 2:
            self.driver_dd.DD_btn(key.release_dd_code)

    def mouse_key_click(self, key: MouseKey, delay: float = 0.1):
        self.mouse_key_press(key)
        time.sleep(delay)
        self.mouse_key_release(key)



    def mouse_move(self, x: int, y: int):
        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(x, y, 0, Mouse.MOUSE_MOVE.press_virtual_code, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.moveR(x, y)
        elif self.mode == 2:
            self.driver_dd.DD_movR(x, y)

    def mouse_scroll(self, times=1):
        if self.mode == 0:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, times * self.WHEEL_DELTA, Mouse.MOUSE_WHEEL.press_virtual_code, 0,
                                ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        elif self.mode == 1:
            self.driver_logi.scroll(times * self.WHEEL_DELTA)
        elif self.mode == 2:
            for i in range(times):
                if times > 0:
                    self.driver_dd.DD_whl(1)
                else:
                    self.driver_dd.DD_whl(2)

if __name__ == '__main__':
    # time.sleep(2)
    # input_key = InputKey()
    # # input_key.mouse_key(Mouse.MOUSE_RIGHT_DOWN)
    # # time.sleep(1)
    # # input_key.mouse_key(Mouse.MOUSE_RIGHT_UP)
    # input_key.mouse_scroll(-1)
    while True:
        time.sleep(1)
        x, y = pyautogui.position()
        print('Mouse position:', x, y)
    pass
