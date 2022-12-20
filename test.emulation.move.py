import ctypes
import multiprocessing
import time
from multiprocessing import Process
from win32gui import GetCursorPos
import pynput
import winsound


end = 'end'
signal = 'signal'
init = {
    end: False,
    signal: False,
}


def keyboard(data):

    def release(key):
        if key == pynput.keyboard.Key.end:
            # 结束程序
            data[end] = True
            winsound.Beep(400, 200)
            return False
        elif key == pynput.keyboard.KeyCode.from_char('f'):
            data[signal] = True

    with pynput.keyboard.Listener(on_release=release) as k:
        k.join()


def consumer(data):

    from toolkit import Timer

    try:
        driver = ctypes.CDLL('logitech.driver.dll')
        ok = driver.device_open() == 1
        if not ok:
            print('初始化失败, 未安装罗技驱动')
    except FileNotFoundError:
        print('初始化失败, 缺少文件')

    def move(x: int, y: int):
        if (x == 0) & (y == 0):
            return
        driver.moveR(x, y, True)

    def x1(x: int):
        move(x, 0)

    def x2(x: int, millis: int):  # 在指定毫秒内在水平方向上移动指定像素
        if x == 0:
            return
        absolute = abs(x)
        direction = x // absolute  # 方向(正负1)
        nanos = millis * 1_000_000
        cost = nanos // absolute  # 每移动一个像素的耗时
        flag = time.perf_counter_ns()
        for i in range(absolute):
            while time.perf_counter_ns() - flag < cost:
                pass
            move(direction, 0)
            flag = time.perf_counter_ns()

    def x3(x: int, millis: int):  # 在指定毫秒内在水平方向上移动指定像素
        if x == 0:
            return
        begin = time.perf_counter_ns()
        absx = abs(x)
        direction = x // absx  # 方向(正负1)
        nanos = millis * 1_000_000  # 毫秒转纳秒
        cost = nanos // absx  # 每移动一个像素的耗时
        flag = time.perf_counter_ns()
        for i in range(absx):
            while time.perf_counter_ns() - flag < cost:
                pass
            move(direction, 0)
            if i == absx - 1:
                return
            flag = time.perf_counter_ns()
            left = nanos - (flag - begin)  # 剩余时间
            cost = left // (absx - i - 1)  # 新的移动一个像素的耗时

    def x4(x: int, millis: int):  # 在指定毫秒内在水平方向上移动指定像素
        begin = time.perf_counter_ns()
        if x == 0:
            return
        times = 0  # 移动次数, 循环判断条件
        absx = abs(x)
        direction = x // absx  # 方向(值只能是正负1)
        nanos = millis * 1_000_000  # 毫秒转纳秒
        while True:  # do-while
            # setup code
            times += 1  # 循环次数的取值范围是[1,100], 101时会break
            # break condition
            if times > absx:
                break
            # loop body
            move(direction, 0)  # 移动方式为: 移动-间隔-移动-间隔-移动-间隔-...-间隔-移动-间隔-移动-间隔-移动, 所以间隔比移动次数少一次
            if times < absx:
                flag = time.perf_counter_ns()
                cost = (nanos - (flag - begin)) // (absx - times)  # 每移动一个像素的耗时
                while time.perf_counter_ns() - flag < cost:
                    pass

    def lowestCommonMultiple(x: int, y: int):  # 最小公倍数
        m, n = x, y
        k = m * n  # k存储两数的乘积
        if m < n:  # 比较两个数的大小，使得m中存储大数，n中存储小数
            temp = m
            m = n
            n = temp
        b = m % n  # b存储m除以n的余数
        while b != 0:
            m = n  # 原来的小数作为下次运算时的大数
            n = b  # 将上一次的余数作为下次相除时的小数
            b = m % n
        result = k // n  # 两数乘积除以最大公约数即为它们的最小公倍数
        return result

    def m(millis: int, horizontal: bool, pixel: int):
        if pixel == 0:
            return
        begin = time.perf_counter_ns()
        times = 0  # 移动次数, 循环判断条件
        absx = abs(pixel)
        direction = pixel // absx  # 方向(值只能是正负1)
        nanos = millis * 1_000_000  # 毫秒转纳秒
        while True:  # do-while
            # setup code
            times += 1  # 循环次数的取值范围是[1,100], 101时会break
            # break condition
            if times > absx:
                break
            # loop body
            # 移动
            if horizontal:
                move(direction, 0)
            else:
                move(0, direction)  # 移动方式为: 移动-间隔-移动-间隔-移动-间隔-...-间隔-移动-间隔-移动-间隔-移动, 所以间隔比移动次数少一次
            # 间隔
            if times < absx:
                flag = time.perf_counter_ns()
                cost = (nanos - (flag - begin)) // (absx - times)  # 每移动一个像素的耗时
                while time.perf_counter_ns() - flag < cost:
                    pass

    def m2(millis: int, x: int, y: int):  # 在指定毫秒内在水平和垂直方向上移动指定像素
        if x == 0 and y == 0:
            return
        elif x == 0 and y != 0:
            m(millis=millis, horizontal=False, pixel=y)
        elif x != 0 and y == 0:
            m(millis=millis, horizontal=True, pixel=x)
        else:
            absx = abs(x)  # 距离的绝对值
            absy = abs(y)
            nanos = millis * 1_000_000  # 毫秒转纳秒
            dirx = x // absx  # direction 方向 (值只能是正负1)
            diry = y // absy
            multiple = lowestCommonMultiple(absx, absy)  # 最小公倍数, 需要将时间分割的段数
            divisorx = multiple // absx
            divisory = multiple // absy
            step = nanos // multiple  # 时间跨度步长, 时间每过一个步长, 都需要判断下该时间点是否需要对两个方向做移动
            for i in range(multiple):
                if i % divisorx == 0:
                    move(dirx, 0)
                if i % divisory == 0:
                    move(0, diry)
                flag = time.perf_counter_ns()
                while time.perf_counter_ns() - flag < step:
                    pass
                # 获取坐标
                # print(GetCursorPos())

    def test2(x):

        t = time.perf_counter_ns()
        x2(x, 1000)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x2(-x, 100)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x2(x, 10)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x2(-x, 1)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

    def test3(x):

        t = time.perf_counter_ns()
        x3(x, 1000)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x3(-x, 100)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x3(x, 10)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x3(-x, 1)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

    def test4(x):

        t = time.perf_counter_ns()
        x4(x, 1000)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x4(-x, 100)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x4(x, 10)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        time.sleep(1)
        t = time.perf_counter_ns()
        x4(-x, 1)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

    def test5(x):

        t = time.perf_counter_ns()
        m2(100, x, 0)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        t = time.perf_counter_ns()
        m2(100, 0, x)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        t = time.perf_counter_ns()
        m2(100, -x, 0)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

        t = time.perf_counter_ns()
        m2(100, 0, -x)
        print(f'{Timer.cost(time.perf_counter_ns() - t)}')

    while True:

        if data[end]:
            break
        elif data[signal]:
            data[signal] = False
            test5(1000)


if __name__ == '__main__':

    data = multiprocessing.Manager().dict()
    data.update(init)
    # 将键鼠监听和压枪放到单独进程中跑
    pk = Process(target=keyboard, args=(data,), name='Keyboard')
    pc = Process(target=consumer, args=(data,), name='Consumer')
    pk.start()
    pc.start()
    pk.join()

