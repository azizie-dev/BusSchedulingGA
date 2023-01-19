from dataclasses import dataclass, asdict, field
from datetime import datetime
import random
import numpy as np
import pprint


@dataclass(frozen=True)
class BusDriver:
    id: str
    name: str = ""


# @dataclass(frozen=True)
# class Bus:
#     id: str


@dataclass(frozen=True)
class TimeSlot:
    id: str
    time: str


# @dataclass(frozen=True)
# class Route:
#     id: str
#     display_name: str


@dataclass
class Schedule:
    def __init__(self):
        self.slots: dict[BusDriver, np.ndarray] = {}
        self.create_data()

    def get_current_date(self):
        self.date = datetime.now()

    def create_data(self):
        TOTAL_DRIVERS = 5

        BUSES = ["BLB7124", "BJE4494"]
        DRIVERS = [f"D{i:02}" for i in range(1, TOTAL_DRIVERS + 1)]
        ROUTES = [
            ("R01", "A"),
            ("R02", "B"),
            ("R03", "C"),
            ("R04", "C2"),
            ("R05", "D"),
            ("R06", "E"),
            ("R07", "G"),
        ]
        TIMES = [
            ("T00", "07:00"),
            ("T01", "08:00"),
            ("T02", "09:00"),
            ("T03", "10:00"),
            ("T04", "11:00"),
            # ("T05", "12:00"),
            # ("T06", "13:00"),
            # ("T07", "14:00"),
            # ("T08", "15:00"),
            # ("T09", "16:00"),
            # ("T10", "17:00"),
            # ("T11", "18:00"),
            # ("T11", "19:00"),
        ]

        # self.buses = [Bus(i) for i in BUSES]
        # self.routes = [Route(id, name) for id, name in ROUTES]
        self.drivers = [BusDriver(i) for i in DRIVERS]
        self.timeslots = [TimeSlot(id, time) for id, time in TIMES]

    def print_data(self):
        # print(self.buses)
        # print(self.routes)
        # print(self.drivers)
        # print(self.timeslots)

        pprint.pprint(self.slots)

    def generate_empty_schedule(self):
        self.slots.update(
            {driver: np.ones(((len(self.timeslots),))) for driver in self.drivers}
        )

    def generate_random_schedule(self, size=5):
        self.slots.update(
            {driver: np.random.randint(0, 2, size) for driver in self.drivers}
        )

    def calc_fitness(self):
        fitness_score = 0

        for driver, times in self.slots.items():
            no_reset = (times == 1).cumsum()
            excess = np.maximum.accumulate((no_reset) * (times == 0))
            result = no_reset - excess

            fitness_score += result

            highest_consecutive_work = result.max()
            if highest_consecutive_work > 3:
                fitness_score = 0

            prefered_break = []
            for i in prefered_break:
                if times[i] == 1:
                    fitness_score += 0.5

        return fitness_score

    def random_mutate(self, mutation_rate=0.1):
        for driver, times in self.slots.items():
            for i in range(times.shape[0]):
                rand = random.random()
                if rand > mutation_rate:
                    times[i] = 1 - times[i]

    def random_crossover(self, other_schedule, size=5):

        data1 = {}
        data2 = {}

        for driver, times1 in self.slots.items():
            times2 = other_schedule[driver]

            assert times1.shape == times2.shape
            random_array = np.random.choice(
                np.arange(0, times1.shape[0]), size, replace=False
            )
            A_new, B_new = self._multi_point_crossover(times1, times2, random_array)

            data1[driver] = A_new
            data2[driver] = B_new

        new_schedule1, new_schedule2 = Schedule(), Schedule()
        new_schedule1.slots.update(data1)
        new_schedule2.slots.update(data2)

        return new_schedule1, new_schedule2

    def _single_point_crossover(self, A, B, x):
        A_new = np.append(A[:x], B[x:])
        B_new = np.append(B[:x], A[x:])

        return A_new, B_new

    def _multi_point_crossover(self, A, B, X):
        for i in X:
            A, B = self._single_point_crossover(A, B, i)
        return A, B


if __name__ == "__main__":
    s1 = Schedule()

    s1.generate_random_schedule()
    print("Before mutation:\n")
    s1.print_data()

    print("After mutation:\n")
    s1.random_mutate(0.5)
    s1.print_data()
