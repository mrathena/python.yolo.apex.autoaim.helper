class Key:
    def __init__(self, char, virtual_code, scan_code, dd_code=0):
        self.char = char
        self.virtual_code = virtual_code
        self.scan_code = scan_code
        self.dd_code = dd_code

    def __repr__(self):
        return f"Key(name='{self.char}', virtual_code={self.virtual_code}, di_scancode={self.scan_code})"


class MouseEvent:
    def __init__(self, char, virtual_code):
        self.char = char
        self.virtual_code = virtual_code

    def __repr__(self):
        return f"MouseEvent(name='{self.char}', virtual_code={self.virtual_code})"


class MouseKey:
    def __init__(self, char, press_virtual_code, release_virtual_code, press_dd_code, release_dd_code, press_logi_code,
                 release_logi_code):
        self.char = char
        self.press_virtual_code = press_virtual_code
        self.release_virtual_code = release_virtual_code
        self.press_dd_code = press_dd_code
        self.release_dd_code = release_dd_code
        self.press_logi_code = press_logi_code
        self.release_logi_code = release_logi_code

    def __repr__(self):
        return f"MouseKey(name='{self.char}', press_virtual_code={self.press_virtual_code}, " \
               f"release_virtual_code={self.release_virtual_code},press_dd_code={self.press_dd_code}," \
               f"release_dd_code={self.release_dd_code})"


class Keyboard:
    Q = Key('Q', 0x51, 0x10, 301)
    W = Key('W', 0x57, 0x11, 302)
    E = Key('E', 0x45, 0x12, 303)
    R = Key('R', 0x52, 0x13, 304)
    T = Key('T', 0x54, 0x14, 305)
    Y = Key('Y', 0x59, 0x15, 306)
    U = Key('U', 0x55, 0x16, 307)
    I = Key('I', 0x49, 0x17, 308)
    O = Key('O', 0x4F, 0x18, 309)
    P = Key('P', 0x50, 0x19, 310)
    A = Key('A', 0x41, 0x1E, 401)
    S = Key('S', 0x53, 0x1F, 402)
    D = Key('D', 0x44, 0x20, 403)
    F = Key('F', 0x46, 0x21, 404)
    G = Key('G', 0x47, 0x22, 405)
    H = Key('H', 0x48, 0x23, 406)
    J = Key('J', 0x4A, 0x24, 407)
    K = Key('K', 0x4B, 0x25, 408)
    L = Key('L', 0x4C, 0x26, 409)
    Z = Key('Z', 0x5A, 0x2C, 501)
    X = Key('X', 0x58, 0x2D, 502)
    C = Key('C', 0x43, 0x2E, 503)
    V = Key('V', 0x56, 0x2F, 504)
    B = Key('B', 0x42, 0x30, 505)
    N = Key('N', 0x4E, 0x31, 506)
    M = Key('M', 0x4D, 0x32, 507)

    # Numbers
    NUM_1 = Key('1', 0x31, 0x02, 201)
    NUM_2 = Key('2', 0x32, 0x03, 202)
    NUM_3 = Key('3', 0x33, 0x04, 203)
    NUM_4 = Key('4', 0x34, 0x05, 204)
    NUM_5 = Key('5', 0x35, 0x06, 205)
    NUM_6 = Key('6', 0x36, 0x07, 206)
    NUM_7 = Key('7', 0x37, 0x08, 207)
    NUM_8 = Key('8', 0x38, 0x09, 208)
    NUM_9 = Key('9', 0x39, 0x0A, 209)
    NUM_0 = Key('0', 0x30, 0x0B, 210)

    # Control keys
    ESCAPE = Key('ESCAPE', 0x1B, 0x01, 100)
    MINUS = Key('MINUS', 0xBD, 0x0C, 211)
    EQUALS = Key('EQUALS', 0xBB, 0x0D, 212)
    BACKSPACE = Key('BACKSPACE', 0x08, 0x0E, 214)
    TAB = Key('TAB', 0x09, 0x0F, 300)
    ENTER = Key('ENTER', 0x0D, 0x1C, 313)
    LCONTROL = Key('LCONTROL', 0xA2, 0x1D, 600)
    RCONTROL = Key('RCONTROL', 0xA3, 0x9D, 607)
    LSHIFT = Key('LSHIFT', 0xA0, 0x2A, 500)
    RSHIFT = Key('RSHIFT', 0xA1, 0x36, 511)
    LALT = Key('LALT', 0xA4, 0x38, 602)
    RALT = Key('RALT', 0xA5, 0xB8, 604)
    LBRACKET = Key('LBRACKET', 0xDB, 0x1A, 311)
    RBRACKET = Key('RBRACKET', 0xDD, 0x1B, 312)
    SEMICOLON = Key('SEMICOLON', 0xBA, 0x27, 410)
    APOSTROPHE = Key('APOSTROPHE', 0xDE, 0x28, 411)
    GRAVE = Key('GRAVE', 0xC0, 0x29, 200)
    BACKSLASH = Key('BACKSLASH', 0xDC, 0x2B, 213)
    COMMA = Key('COMMA', 0xBC, 0x33, 508)
    PERIOD = Key('PERIOD', 0xBE, 0x34, 509)
    SLASH = Key('SLASH', 0xBF, 0x35, 510)
    CAPSLOCK = Key('CAPSLOCK', 0x14, 0x3A)
    F1 = Key('F1', 0x70, 0x3B, 101)
    F2 = Key('F2', 0x71, 0x3C, 102)
    F3 = Key('F3', 0x72, 0x3D, 103)
    F4 = Key('F4', 0x73, 0x3E, 104)
    F5 = Key('F5', 0x74, 0x3F, 105)
    F6 = Key('F6', 0x75, 0x40, 106)
    F7 = Key('F7', 0x76, 0x41, 107)
    F8 = Key('F8', 0x77, 0x42, 108)
    F9 = Key('F9', 0x78, 0x43, 109)
    F10 = Key('F10', 0x79, 0x44, 110)
    F11 = Key('F11', 0x7A, 0x57, 111)
    F12 = Key('F12', 0x7B, 0x58, 112)
    NUM_LOCK = Key('NUMLOCK', 0x90, 0x45, 810)
    SCROLL_LOCK = Key('SCROLLLOCK', 0x91, 0x46, 701)
    LWIN = Key('LWIN', 0x5B, 0xDB, 601)
    RWIN = Key('RWIN', 0x5C, 0xDC, 605)
    SPACE = Key('SPACE', 0x20, 0x39, 603)

    # Arrow keys
    UP_ARROW = Key('UP', 0x26, 0xC8, 709)
    DOWN_ARROW = Key('DOWN', 0x28, 0xD0, 711)
    LEFT_ARROW = Key('LEFT', 0x25, 0xCB, 710)
    RIGHT_ARROW = Key('RIGHT', 0x27, 0xCD, 712)
    # Control keys above arrow keys
    INSERT = Key('INSERT', 0x2D, 0xD2, 703)
    HOME = Key('HOME', 0x24, 0xC7, 704)
    PAGE_UP = Key('PAGE_UP', 0x21, 0xC9, 705)
    DELETE = Key('DELETE', 0x2E, 0xD3, 706)
    END = Key('END', 0x23, 0xCF, 707)
    PAGE_DOWN = Key('PAGE_DOWN', 0x22, 0xD1, 708)
    # Numpad
    NUMPAD_0 = Key('NUMPAD_0', 0x60, 0x52, 800)
    NUMPAD_1 = Key('NUMPAD_1', 0x61, 0x4F, 801)
    NUMPAD_2 = Key('NUMPAD_2', 0x62, 0x50, 802)
    NUMPAD_3 = Key('NUMPAD_3', 0x63, 0x51, 803)
    NUMPAD_4 = Key('NUMPAD_4', 0x64, 0x4B, 804)
    NUMPAD_5 = Key('NUMPAD_5', 0x65, 0x4C, 805)
    NUMPAD_6 = Key('NUMPAD_6', 0x66, 0x4D, 806)
    NUMPAD_7 = Key('NUMPAD_7', 0x67, 0x47, 807)
    NUMPAD_8 = Key('NUMPAD_8', 0x68, 0x48, 808)
    NUMPAD_9 = Key('NUMPAD_9', 0x69, 0x49, 809)
    NUMPAD_MULTIPLY = Key('NUMPAD_MULTIPLY', 0x6A, 0x37, 812)
    NUMPAD_ADD = Key('NUMPAD_ADD', 0x6B, 0x4E, 814)
    NUMPAD_ENTER = Key('NUMPAD_ENTER', 0xD, 0x9C, 815)
    NUMPAD_SUBTRACT = Key('NUMPAD_SUBTRACT', 0x6D, 0x4A, 813)
    NUMPAD_DECIMAL = Key('NUMPAD_DECIMAL', 0x6E, 0x53, 816)
    NUMPAD_DIVIDE = Key('NUMPAD_DIVIDE', 0x6F, 0xB5, 811)


class Mouse:
    MOUSE_LEFT = MouseKey('Mouse Left', 0x0002, 0x0004, 1, 2, 1, 1)
    MOUSE_RIGHT = MouseKey('Mouse Right', 0x0008, 0x0010, 4, 8, 2, 2)
    MOUSE_MIDDLE = MouseKey('Mouse Middle', 0x0020, 0x0040, 16, 32, 3, 3)
    MOUSE_X = MouseKey('Mouse X', 0x0080, 0x0100, 64, 128, 4, 4)
    MOUSE_WHEEL = MouseKey('Mouse Wheel', 0x0800, 0x0000, 0, 0, 0, 0)
    MOUSE_MOVE = MouseKey('Mouse Move', 0x0001, 0x0000, 0, 0, 0, 0)
