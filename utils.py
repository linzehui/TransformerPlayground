
from datetime import timedelta
import time

start_time=time.time()
def get_time_dif(start_time):
    """get time used"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

