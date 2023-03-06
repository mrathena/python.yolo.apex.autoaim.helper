
from win32gui import EnumWindows, GetWindowText, GetClassName

title = 'Apex Legends'

windowHandleList = []
EnumWindows(lambda hwnd, param: param.append(hwnd), windowHandleList)
filteredWindowHandleList = []
for hwnd in windowHandleList:
    if title in GetWindowText(hwnd):
        filteredWindowHandleList.append(hwnd)
size = len(filteredWindowHandleList)
if size == 0:
    print('未找到窗体')
else:
    message = '找到的窗体如下'
    for i, hwnd in enumerate(filteredWindowHandleList):
        message += f'\r\n\t{i + 1}: {hwnd}, {GetWindowText(hwnd)}, {GetClassName(hwnd)}'
    print(message)










