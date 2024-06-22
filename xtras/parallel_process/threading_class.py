import threading
import time


class ReadWriteThread:

    def __init__(self) -> None:

        self.val1 = 0.0
        self.val2 = 0.0
        self.vt1 = threading.Thread(target=self.update_val1)
        self.vt2 = threading.Thread(target=self.update_val2)
        try:
            self.vt1.start()
            self.vt2.start()
        except KeyboardInterrupt:
            self.vt1.join()
            self.vt2.join()
        finally:
            print("finished")


    def update_val1(self):
        while True:
            self.val1 += 1
            print(f"> self.val1: {self.val1}")

    def update_val2(self):
        while True:
            self.val2 += 1
            print(f"> self.val2: {self.val2}")


if __name__ == "__main__":
    tr = ReadWriteThread()
