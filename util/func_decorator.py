import time

def interrupt_decorator(sideFunction):
    def decorator(mainFunction):
        def wrapper(*args, **kwargs):
            try:
                mainFunction(*args, **kwargs)
            except KeyboardInterrupt:
                sideFunction()
        return wrapper
    return decorator

def sideFunc():
    print("exiting")

@interrupt_decorator(sideFunc)
def mainFunc():
    print("running...")
    time.sleep(2)
    print("finished")

mainFunc()