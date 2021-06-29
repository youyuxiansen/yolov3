from collections import OrderedDict
import threading
import time

asd = OrderedDict()
lock = threading.Lock()

asd[2] = 1
asd[3] = 1
asd[4] = 1
asd[1] = 1


def insert(dict1):
	lock.acquire()
	dict1[10] = 1
	print('insert finished')
	time.sleep(3)
	lock.release()


def pop(dict1):
	lock.acquire()
	dict1.pop(3)
	print('pop finished')
	time.sleep(3)
	lock.release()


thread1 = threading.Thread(target=insert, args=(asd,), name='thread1')
thread2 = threading.Thread(target=pop, args=(asd,), name='thread2')
thread1.start()
thread2.start()
print("thread finished...exiting")
