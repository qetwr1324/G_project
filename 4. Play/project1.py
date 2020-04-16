import time

import pyvjoy


def test():
    j = pyvjoy.VJoyDevice(1)
    i = 1
    while (True):
        # left_stick
        j.data.wAxisX = i
        # left_trigger
        j.data.wAxisY = i
        # right_trigger
        j.data.wAxisZ = i
        j.update()
        print(i)
        i = i + 1
        time.sleep(0.0001)


if __name__ == "__main__":
    test()
