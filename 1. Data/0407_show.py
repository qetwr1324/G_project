import time
from datetime import datetime
import xinput
import os
from operator import itemgetter, attrgetter
from PIL import ImageGrab
import sys
import pandas as pd

left_speed = 0
right_speed = 0
wheel = 0
df = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)
def save():
    print("Done")
    now = time.localtime()
    folder_name = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(os.path.join(folder_name))
    try:
        if not (os.path.isdir('./'+str(folder_name))):
            os.makedirs(os.path.join(str(folder_name)))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    joysticks = xinput.XInputJoystick.enumerate_devices()
    device_numbers = list(map(attrgetter('device_number'), joysticks))
    print('found %d devices: %s' % (len(joysticks), device_numbers))
    if not joysticks:
        sys.exit(0)
    j = joysticks[0]
    print('using %d' % j.device_number)
    battery = j.get_battery_information()
    @j.event
    def on_button(button, pressed):
        print('button', button, pressed)
    left_speed = 0
    right_speed = 0
    @j.event
    def on_axis(axis, value):
        global left_speed
        global right_speed
        global wheel
        global df
        if axis == "l_thumb_y":
            pass
        else:
            now = datetime.now()
            file_name = now.strftime("%Y%m%d%H%M%S%f.jpg")
            if axis == "l_thumb_x":
                wheel = value
            if axis == "left_trigger":
                left_speed = value
            elif axis == "right_trigger":
                right_speed = value
            data = pd.DataFrame({"file_name" : [file_name] , "break" : [left_speed], "accel" : [right_speed], "wheel" : [wheel]})
            img=ImageGrab.grab()
            img.save("./"+folder_name+"/"+file_name)
            df= pd.concat([df,data],ignore_index=True)
            print(df)
    while True:
        j.dispatch_events()
        time.sleep(.0)
        if df.index.size % 1000 == 999:
            df.to_csv("./"+folder_name+"/"+folder_name+".csv")
            break


if __name__ == "__main__":
    time.sleep(10)
    while True:
        df = pd.DataFrame([], columns=["file_name", "break", "accel", "wheel"], index=None)
        save()