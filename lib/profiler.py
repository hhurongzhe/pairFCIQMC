import time


class Profiler:
    def __init__(self):
        self.profiler = dict()

    def add_timing(self, tag: str, time: float):
        if tag in self.profiler.keys():
            self.profiler[tag] += time
        else:
            self.profiler[tag] = time

    def print_timings(self):
        print("Overview of timings \n")

        print("{:40}".format("Code section"), "runtime [s]    percentage [%]")
        print("---------------------------------------------------------------------")
        total = 0.0
        for k, v in self.profiler.items():
            total += v

        for k, v in self.profiler.items():
            print("{:40}".format(k), "%3.4f" % v, "       ", "%2.3f" % (100.0 * v / total))

        print("---------------------------------------------------------------------")
        print("{:40}".format("Total"), "%3.4f" % total)
