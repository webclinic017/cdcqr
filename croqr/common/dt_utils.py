import math
import time
from datetime import datetime
from time import mktime


class DataTimeUtil:
    def __init__(self):
        pass

    @staticmethod
    def epoch_time_converter(epoch_time):
        """
        Returns the parsed time compoents from epoch time upto milisecond

        Returns:
                dt (datetime)  up to second
                microsecond (int) up to microsecond
        """
        dt = epoch_time // 1e6
        microsecond = int(epoch_time - dt * 1e6)
        struct_time = time.localtime(dt)
        dt1 = datetime.fromtimestamp(mktime(struct_time))

        return dt1, microsecond

    @staticmethod
    def ceil_dt(delta, dt):
        ref_dt = datetime(1970, 1, 1)
        return ref_dt + math.ceil((dt - ref_dt) / delta) * delta
