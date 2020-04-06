import time
from pynput.mouse import Controller
from pynput.keyboard import Listener, Key, KeyCode
 
store = set()
 
HOT_KEYS = {
    'print_hello': set([ Key.alt_l, KeyCode(char='1')] )
}
 
def print_hello():
   while(True):
    print("Current position: " + str(Controller().position))
    time.sleep(0.1)
 
def handleKeyPress( key ):
    store.add( key )
 
    for action, trigger in HOT_KEYS.items():
        CHECK = all([ True if triggerKey in store else False for triggerKey in trigger ])
 
        if CHECK:
            try:
                func = eval( action )
                if callable( func ):
                   func()
            except NameError as err:
                print( err )
 
def handleKeyRelease( key ):
    if key in store:
        store.remove( key )
        
    # 종료
    if key == Key.esc:
        return False
 
with Listener(on_press=handleKeyPress, on_release=handleKeyRelease) as listener:
    listener.join()
