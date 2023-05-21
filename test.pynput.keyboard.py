import multiprocessing
from multiprocessing import Process
from pynput.keyboard import Key, KeyCode, Listener


def keyboard():

    def release(key):
        print(key)
        if key == Key.end:
            # 结束程序
            return False
        elif hasattr(key, 'vk'):
            if key.vk == 103:
                print(103)
            elif key.vk == 104:
                print('i')
            elif key.vk == 105:
                print('d')
            elif key == KeyCode.from_char('P'):
                print('P')
            elif key == KeyCode.from_char('I'):
                print('I')
            elif key == KeyCode.from_char('D'):
                print('D')

    with Listener(on_release=release) as k:
        k.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    pk = Process(target=keyboard, args=(), name='Keyboard')
    pk.start()
    pk.join()
