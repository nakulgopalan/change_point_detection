from pynput import mouse
import numpy as np


global count
count = 0
global trajectory, position_array
position_array = []

# collects continuous data based on mouse movement. Starts at a click so can change windows
# stop the deamon after a second click!


def on_move(x, y):
    global position_array
    print('Pointer moved to {0}'.format(
        (x, y)))
    if count>=1:
        position_array.append([x,y])



def on_click(x, y, button, pressed):
    global count
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    count+=1
    print(count)
    if (not pressed) & (count>=3):
        # Stop listener
        return False

def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))

# Collect events until released
with mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll) as listener:
    listener.join()

np.save('test_0', np.asarray(position_array))
print("finished!!")