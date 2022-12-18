
# 说明

> [CSDN Python Apex YOLO V5 6.2 目标检测与自瞄 全过程记录](https://blog.csdn.net/mrathena/article/details/126860226)

因为没有计算机视觉相关方向的专业知识, 所以做出来的东西, 有一定效果, 但是还有很多不足

因为 CSDN 部分规则原因, CSDN 文章将不能再更新, 如果后续代码有优化, 只会在 GitHub 同步

源码说明:
- apex.py: 主文件, 自瞄逻辑与程序控制都在这里
- toolkit.py: 自行封装的工具, 封装了截图推理等工具
- apex.new.py: 主文件, 用于测试新想法
- toolkit2.py: 工具包, 用于测试新想法
- apex.fov.py: 使用 FOV 的主文件, 已放弃 FOV, 感觉没啥用, 一个倍数完全可以代替
- weights.*.pt: 训练好的模型权重文件, 大概训练了下, 识别不一定精准
- logitech.driver.dll: 大佬封装的调用罗技驱动的库文件
- logitech.test.py: 用于测试罗技驱动安装配置是否正确
- detect.realtime.py: 实时展示推理结果, 用于测试权重文件是否好用, 可通过播放Apex集锦来测试
- grab.*: 截图工具, 用于准备数据集
- label.*: 标注工具, 用于使用现有模型标注数据集, 完了再手动修正
- dataset.*.yaml: 训练模型时的数据集配置文件
- train.*: 训练工具, 用于训练模型
- detect.*: 推理工具, 用于推理目标检测
- export.*: 将 .pt 导出为 .engine

参数说明: apex.py 中的 init
- ads: 就是一个作用于鼠标移动距离的倍数, 用于调整移动鼠标时的实际像素和鼠标 eDPI 的关系. 
  - 开启自瞄后, 不断瞄准目标旁边并按住 F 键, 当准星移动稳定且精准快速不振荡时, 就找到了合适的 ADS 值
- classes: --

按键说明:
- End: 退出程序, 全局有效
- Down / 鼠标侧上键: 自瞄开关(默认关), 仅游戏中有效
- Up: 显示开关(默认关), 用于弹窗显示识别结果, 仅游戏中有效
- F / 鼠标左键(按下时生效): 执行目标识别与自瞄, 打开自瞄开关后有效, 仅游戏中有效
- Right: 是否瞄头(默认关), 有3种方式, 1:根据识别结果中的头部瞄准, 2:根据识别到的身体推测头部位置, 3:混合1和2(优先1). 这里只使用了2, 其他可自行实现
- Left: 是否预瞄(默认关): 感觉效果不是很好
- PageDown: 是否仿真(默认开): 仿真时, 水平垂直方向的移动力度会减小, 根据参数 horizontal 和 vertical 来定
- PageUp: 仿真时是否加随机移动(默认关): 仿真时如果开了加随机值, 则左右会有一定概率和幅度的摇动

其他说明:
- 游戏需要限制帧数, 以便给显卡让出足够算力做目标检测, 比如锁60帧 `+fps_max 60`, 根据自己的情况定

# 环境准备

需要 PyTorch Cuda 环境, 参考 [文章](https://blog.csdn.net/mrathena/article/details/126860226) 中 `环境准备 YOLO V5 6.2` 部分

## 操纵键鼠

大多FPS游戏都屏蔽了操作鼠标的Win函数, 要想在游戏中用代码操作鼠标, 需要一些特殊的办法, 其中罗技驱动算是最简单方便的了

代码直接控制罗技驱动向操作系统(游戏)发送鼠标命令, 达到了模拟鼠标操作的效果, 这种方式是鼠标无关的, 任何鼠标都可以实现

我们不会直接调用罗技驱动, 但是有大佬已经搭过桥了, 有现成的调用驱动的dll, 只是需要安装指定版本的罗技驱动配合才行

### 驱动安装和系统设置

> [百度网盘 罗技键鼠驱动](https://pan.baidu.com/s/1VkE2FQrNEOOkW6tCOLZ-kw?pwd=yh3s)

罗技驱动分 LGS (老) 和 GHub (新), LGS 的话, 需要使用 9.02.65 版本的, GHub 的话, 需要使用 2021.11 之前的, 二者自选其一即可

装好驱动后, 无需重启电脑. 运行 `屏蔽GHUB更新.exe` 防止更新

另外需要确保 控制面板-鼠标-指针选项 中下面两个设置
- 提高指针精确度 选项去掉, 不然会造成实际移动距离变大
- 选择指针移动速度 要在正中间, 靠右会导致实际移动距离过大, 靠左会导致指针移动距离过小

运行 `logitech.test.py` 查看效果, 确认安装是否成功

### 代码

大佬封装的 `logitech.driver.dll` 没有文档, 下面是某老哥列出的该库里面的方法, 具体用法参考 `logitech.test.py`

![](https://github.com/mrathena/python.apex.weapon.auto.recognize.and.suppress/blob/master/readme/20221204.131618.213.png)

## 键鼠监听

> [Pynput 说明](https://pypi.org/project/pynput/)

注意调试回调方法的时候, 不要打断点, 不然会卡死IO, 导致鼠标键盘失效

回调方法如果返回 False, 监听线程就会自动结束, 所以不要随便返回 False

键盘的特殊按键采用 `keyboard.Key.tab` 这种写法，普通按键用 `keyboard.KeyCode.from_char('c')` 这种写法, 有些键不知道该怎么写, 可以 `print(key)` 查看信息

> 钩子函数本身是阻塞的。也就是说钩子函数在执行的过程中，用户正常的键盘/鼠标操作是无法输入的。所以在钩子函数里面必须写成有限的操作（即O(1)时间复杂度的代码），也就是说像背包内配件及枪械识别，还有下文会讲到的鼠标压枪这类时间开销比较大或者持续时间长的操作，都不适合写在钩子函数里面。这也解释了为什么在检测到Tab（打开背包）、鼠标左键按下时，为什么只是改变信号量，然后把这些任务丢给别的进程去做的原因。

# 扩展

## Python Apex 武器自动识别与压枪

> [GitHub python.apex.weapon.auto.recognize.and.suppress](https://github.com/mrathena/python.apex.weapon.auto.recognize.and.suppress)
> 
## Python Pubg 武器自动识别与压枪

> [GitHub python.pubg.weapon.auto.recognize.and.suppress](https://github.com/mrathena/python.pubg.weapon.auto.recognize.and.suppress)

# 拓展 通用型人体骨骼检测 与 自瞄, 训练一次, FPS 游戏通用

> [【亦】警惕AI外挂！我写了一个枪枪爆头的视觉AI，又亲手“杀死”了它](https://www.bilibili.com/video/BV1Lq4y1M7E2/)

> [YOLO V7 keypoint 人体关键点检测](https://xugaoxiang.com/2022/07/21/yolov7/)

大多数 FPS 游戏中要检测的目标都为人形, 可以训练一个 通用型人体骨骼检测模型, 在类似游戏中应该有不错的效果
