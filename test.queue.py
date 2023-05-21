from queue import Queue

queue = Queue(maxsize=3)

queue.put(1)
queue.put(2)
queue.put(3)
if queue.full:
    print(queue.get())
    queue.put(4)

print(list(queue.queue))
print(queue.full())
