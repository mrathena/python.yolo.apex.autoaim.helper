import os

from toolkit import Detector

detector = Detector('model.for.apex.2.engine')

directory = r'D:\resource\develop\python\dataset.yolo\apex\action\data'
files = os.listdir(directory)
print(f'total files: {len(files)}')
paths = []
for file in files:
    path = os.path.join(directory, file)
    if path.endswith('.txt'):
        continue
    paths.append(path)
print(f'image files: {len(paths)}')

for i, path in enumerate(paths):
    print(f'{i + 1}/{len(paths)}')
    detector.label(path)
