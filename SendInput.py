import ctypes


# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-hardwareinput
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-input
# https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

# ScanCode
# https://github.com/Lateralus138/Key-ScanCode/releases/tag/1.9.30.18


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),  # 水平方向的绝对位置/相对移动量(像素), dwFlags 中包含 MOUSEEVENTF_ABSOLUTE 标识就是绝对移动, 否则是相对移动
                ("dy", ctypes.c_long),  # 垂直方向的绝对位置/相对移动量(像素)
                ("mouseData", ctypes.c_ulong),  # 某些事件的额外参数, 如: MOUSEEVENTF_WHEEL(中键滚动), 可填正负值, 一个滚动单位是120(像素?); 还有 MOUSEEVENTF_XDOWN/MOUSEEVENTF_XUP
                ("dwFlags", ctypes.c_ulong),  # 事件标识集, 可以是移动或点击事件的合理组合, 即可以一个命令实现移动且点击
                ("time", ctypes.c_ulong),  # 事件发生的时间戳, 可以指定发生的时间? 传入0则使用系统提供的时间戳
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]  # 应用可通过 GetMessageExtraInfo 来接收通过此参数传递的额外消息


class KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),  # 虚拟键码, 范围在[1,254], 如果dwFlags指定了KEYEVENTF_UNICODE, 则wVk必须是0, https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
                ("wScan", ctypes.c_ushort),  # 键的硬件扫描码, 如果dwFlags指定了KEYEVENTF_UNICODE, 则wScan需为一个Unicode字符
                ("dwFlags", ctypes.c_ulong),  # 事件标识集, 可合理组合
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class Inner(ctypes.Union):  # 共用体, 和结构体类似, 但是各成员属性共用同一块内存空间, 实例化时的空间大小就是成员属性中最大的那个的空间大小, 实例化时只能赋值ki/mi/hi中的一个
    _fields_ = [("ki", KeyboardInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),  # 输入事件类型
                ("ii", Inner)]


"""
MouseInput
"""
MOUSEEVENTF_MOVE = 0x0001  # 移动
MOUSEEVENTF_LEFTDOWN = 0x0002  # 左键按下
MOUSEEVENTF_LEFTUP = 0x0004  # 左键释放
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080  # 侧键按下
MOUSEEVENTF_XUP = 0x0100  # 侧键释放
MOUSEEVENTF_WHEEL = 0x0800  # 滚轮垂直滚动, mouseData需传入滚动值, 一个滚动单位是120(像素?)
MOUSEEVENTF_HWHEEL = 0x1000  # 滚轮水平滚动
MOUSEEVENTF_MOVE_NOCOALESCE = 0x2000  # 移动消息不会被合并
MOUSEEVENTF_VIRTUALDESK = 0x4000  # Maps coordinates to the entire desktop. Must be used with MOUSEEVENTF_ABSOLUTE.
MOUSEEVENTF_ABSOLUTE = 0x8000  # 鼠标移动事件, 如果设置此标记就是绝对移动, 否则就是相对移动. 相对鼠标运动受鼠标速度(控制面板中的指针移动速度)和两个鼠标阈值(?和速度在同一个地方设置)的影响, 绝对值移动时范围是[0,65535]

XBUTTON1 = 0x0001  # 侧下键, dwFlags为MOUSEEVENTF_XDOWN/MOUSEEVENTF_XUP时, mouseData需传入x键值
XBUTTON2 = 0x0002  # 侧上键

WHEEL_DELTA = 120  # 鼠标滚轮滚动一个单位的最小值

# dwFlags 不能同时包含 MOUSEEVENTF_WHEEL/MOUSEEVENTF_XDOWN/MOUSEEVENTF_XUP, 因为它们都需要使用mouseData字段. 如果不包含这3个字段, 则mouseData应该传入0
# dwFlags 仅仅能表示状态的变化, 而不能表示状态的持续, 如它能表示按下左键, 但按下左键不代表持续按住左键, 但我测下来好像可以(只发送down不发送up则相当于一直按住)?
# If dwFlags contains MOUSEEVENTF_WHEEL, then mouseData specifies the amount of wheel movement. A positive value indicates that the wheel was rotated forward, away from the user; a negative value indicates that the wheel was rotated backward, toward the user. One wheel click is defined as WHEEL_DELTA, which is 120.

"""
KeyboardInput
"""
KEYEVENTF_EXTENDEDKEY = 0x0001  # If specified, the scan code was preceded by a prefix byte that has the value 0xE0 (224).
KEYEVENTF_KEYUP = 0x0002  # 指定该标记, 就是按键弹起, 否则就是按键按下
KEYEVENTF_SCANCODE = 0x0008  # 指定该标记, 就是使用硬件扫描码来发送按键按下事件, wVk参数将被忽略
KEYEVENTF_UNICODE = 0x0004  # 发送Unicode字符消息(按下), 可以与KEYEVENTF_KEYUP组合使用

# 按键的虚拟键码和扫描代码是两套码, 虚拟键码可能因键盘/布局不同而不同, 但扫描代码是键盘无关的, 所以推荐使用扫描码
# dwFlags 仅仅能表示状态的变化, 而不能表示状态的持续, 如按下w键, 物理键盘会持续输入w, 而模拟键盘则仅能输入一个w

"""
Input
"""
INPUT_MOUSE = 0  # 鼠标输入事件
INPUT_KEYBOARD = 1  # 键盘输入事件
INPUT_HARDWARE = 2  # 硬件输入事件

"""
SendInput
cInputs: pInputs 数组的个数, 即SendInput可以一下发好几个鼠标事件/键盘事件, 这些事件存储在一个连续的数组空间里
pInputs: 输入事件的实例的指针
cbSize: 每一个INPUT事件的结构体空间, 鼠标事件和键盘事件应该不能同时放到pInputs数组中, 因为他们的size不同 
该函数返回成功插入键盘或鼠标输入流的事件数。如果函数返回零，则输入已被另一个线程阻塞。要获取扩展错误信息，请调用GetLastError.
"""


def SendInput(*inputs):  # 接收任意个参数, 将其打包成为元组形参, 双*是打包成为字典形参
    nInputs = len(inputs)
    pointer = Input * nInputs
    pInputs = pointer(*inputs)
    # 创建一个指定类型的C数组并返回首指针, 格式如下
    # pointer = (Type * 数组长度)(填充该数组的实例,用逗号分隔开,个数不超过数组长度)
    # 1. pointer = (Input * 3)(); 创建一个空间为3个Input的C数组, 其中的每个位置的Input已经按默认值初始化
    # 2. pointer = (Input * 3)(input); 创建一个空间为3个Input的C数组, 并将input赋值给第一个位置
    # 3. pointer = (Input * 3)(input, input2, input2), 创建一个空间为3个Input的C数组, 并将input赋值给第一个位置, 将input2赋值给第二第三个位置
    # 4. pointer = (Input * 3)(input, input2, input3, input4), 创建一个空间为3个Input的C数组, 赋值时报错, 因为没有放input4的空间
    # 5. pointer = (Input * 3)(*inputs); 创建一个空间为3个Input的C数组, 并将集合类型的inputs解包并赋值给数组的对应位置
    # 6. 也可以如上分开两行写
    cbSize = ctypes.sizeof(Input)
    return ctypes.windll.user32.SendInput(nInputs, pInputs, cbSize)


class Keyboard:

    @staticmethod
    def press(wVk):  # 十六进制虚拟键码, 范围在[1,254], 如果dwFlags指定了KEYEVENTF_UNICODE, 则wVk必须是0, https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(wVk, 0, 0, 0, None))))

    @staticmethod
    def release(wVk):
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(wVk, 0, KEYEVENTF_KEYUP, 0, None))))

    @staticmethod
    def pressByScanCode(wScan):  # 十六进制键扫描码, 可通过该工具获得, https://github.com/Lateralus138/Key-ScanCode/releases/tag/1.9.30.18
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(0, wScan, KEYEVENTF_SCANCODE, 0, None))))

    @staticmethod
    def releaseByScanCode(wScan):
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(0, wScan, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, None))))

    @staticmethod
    def pressByUnicode(wScan):  # 十六进制Unicode编码, 比如 汉字 "一" 的十六进制UTF8编码为 0x4e00, 则可通过发送该编码直接输入汉字一
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(0, wScan, KEYEVENTF_UNICODE, 0, None))))

    @staticmethod
    def releaseByUnicode(wScan):
        return SendInput(Input(INPUT_KEYBOARD, Inner(ki=KeyboardInput(0, wScan, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None))))


class Mouse:

    @staticmethod
    def leftDown():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None))))

    @staticmethod
    def leftUp():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None))))

    @staticmethod
    def leftClick():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, None))))

    @staticmethod
    def leftDoubleClick():
        event = Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, None)))
        return SendInput(event, event)

    @staticmethod
    def rightDown():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, None))))

    @staticmethod
    def rightUp():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, None))))

    @staticmethod
    def rightClick():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_RIGHTUP, 0, None))))

    @staticmethod
    def middleDown():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_MIDDLEDOWN, 0, None))))

    @staticmethod
    def middleUp():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_MIDDLEUP, 0, None))))

    @staticmethod
    def middleClick():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, 0, MOUSEEVENTF_MIDDLEDOWN | MOUSEEVENTF_MIDDLEUP, 0, None))))

    @staticmethod
    def x1Down():  # 侧下键
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON1, MOUSEEVENTF_XDOWN, 0, None))))

    @staticmethod
    def x1Up():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON1, MOUSEEVENTF_XUP, 0, None))))

    @staticmethod
    def x1Click():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON1, MOUSEEVENTF_XDOWN | MOUSEEVENTF_XUP, 0, None))))

    @staticmethod
    def x2Down():  # 侧上键
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON2, MOUSEEVENTF_XDOWN, 0, None))))

    @staticmethod
    def x2Up():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON2, MOUSEEVENTF_XUP, 0, None))))

    @staticmethod
    def x2Click():
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, XBUTTON2, MOUSEEVENTF_XDOWN | MOUSEEVENTF_XUP, 0, None))))

    @staticmethod
    def move(x, y, absolute=False):
        if not absolute:
            return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(x, y, 0, MOUSEEVENTF_MOVE, 0, None))))
        else:
            # 绝对值移动时, 屏幕宽高的取值范围都是[0, 65535], 需要自行换算一下
            # 获取显示分辨率(非物理分辨率)
            w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
            rx, ry = int(x * 65535 / w), int(y * 65535 / h)
            return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(rx, ry, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, None))))

    @staticmethod
    def scroll(delta, vertical=True):  # 滚动delta个单位, 一个单位是120(像素?), 区分正负值, 正值:向下转/向右转, 负值:向上移/向左转
        dwFlags = MOUSEEVENTF_WHEEL if vertical else MOUSEEVENTF_HWHEEL
        return SendInput(Input(INPUT_MOUSE, Inner(mi=MouseInput(0, 0, delta * WHEEL_DELTA, dwFlags, 0, None))))


if __name__ == '__main__':
    print(Keyboard.press(0x51))
    print(Keyboard.pressByScanCode(0x1e))
    print(Keyboard.pressByUnicode(0x4e00))
