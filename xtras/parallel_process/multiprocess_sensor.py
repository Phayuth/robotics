import multiprocessing
import time


class SensorProcess(multiprocessing.Process):
    def __init__(self, sensor_id, update_interval):
        super().__init__()
        self.sensor_id = sensor_id
        self.update_interval = update_interval
        self.current_value = multiprocessing.Value("d", 0.0)  # Shared value for multiprocessing

    def run(self):
        while True:
            # Simulate reading sensor data and updating value
            new_value = self.read_sensor_data()
            with self.current_value.get_lock():
                self.current_value.value = new_value
            time.sleep(self.update_interval)

    def read_sensor_data(self):
        # Simulate reading sensor data
        # In a real application, this method would read actual sensor data
        # For demonstration purposes, it generates a dummy value
        return self.current_value.value + 10.0  # Dummy value based on sensor id

    def get_current_value(self):
        with self.current_value.get_lock():
            return self.current_value.value


if __name__ == "__main__":
    num_sensors = 1
    update_interval = 0.1  # in seconds

    sensor_processes = []

    # Create and start sensor processes
    for i in range(num_sensors):
        sensor_process = SensorProcess(sensor_id=i + 1, update_interval=update_interval)
        sensor_process.start()
        sensor_processes.append(sensor_process)

    try:
        while True:
            # Main program loop
            # Print current values from each sensor process
            for sensor_process in sensor_processes:
                print(f"Sensor {sensor_process.sensor_id}: {sensor_process.get_current_value()}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        # Terminate sensor processes on keyboard interrupt
        for sensor_process in sensor_processes:
            sensor_process.terminate()
            sensor_process.join()
