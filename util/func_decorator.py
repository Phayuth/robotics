import time

def interrupt_decorator(handler):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            try:
                fun(*args, **kwargs)
            except KeyboardInterrupt:
                handler()
        return wrapper
    return decorator

# def fun():
#     try:
#         print("running...")
#         time.sleep(30)
#         print("finished")
#     except KeyboardInterrupt:
#         print("exiting...")

@interrupt_decorator(lambda: print("exiting"))
def fun2():
    print("running...")
    time.sleep(2)
    print("finished")

fun2()