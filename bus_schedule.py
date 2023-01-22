from dataclasses import dataclass, asdict, field
from datetime import datetime
import random
import numpy as np
import pprint
from itertools import groupby


@dataclass(frozen=True)
class BusDriver:
    id: str
    name: str = ""


@dataclass(frozen=True)
class TimeSlot:
    id: str
    time: str


@dataclass
class Schedule:
    def __init__(self):
        self.slots: dict[BusDriver, np.ndarray] = {}
        self.create_data()

    def get_current_date(self):
        self.date = datetime.now()

    def create_data(self):
        TOTAL_DRIVERS = 23
        DRIVERS = [f"D{i:02}" for i in range(1, TOTAL_DRIVERS + 1)]

        TIMES = [
            ("T" + "{:02d}".format(i), "{:02d}:{:02d}".format(i // 2 + 7, (i % 2) * 30))
            for i in range(31)
        ]

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

    def generate_random_schedule(self):
        new_schedule = {}

        for driver in self.drivers:
            new_times = self._generate_raw_times()
            while not (self._is_valid_times(new_times)):
                new_times = self._generate_raw_times()

            new_schedule[driver] = new_times

        self.slots.update(new_schedule)

    def _generate_raw_times(self):
        new_times = np.ones(len(self.timeslots))
        random_indices = np.random.choice(
            np.arange(0, len(self.timeslots) - 1, 2), 6, replace=False
        )
        # random_indices = np.concatenate((random_indices, random_indices + 1), axis=0)
        new_times[random_indices] = 0

        return new_times

    def _is_valid_times(self, arr, overtime_idx=23):

        mand_work_count = np.sum(arr[:overtime_idx] == 1)
        over_work_count = np.sum(arr[overtime_idx:] == 1)
        break_count = np.sum(arr == 0)

        consecutive_count = [
            len(list(group)) for key, group in groupby(arr) if key == 1
        ]

        max_consecutive_work = max(consecutive_count)
        min_consecutive_work = min(consecutive_count)

        max_consecutive_break = min(
            [len(list(group)) for key, group in groupby(arr) if key == 0]
        )

        # Hard Constraint 1: The mandatory working period for every driver is 8 hours per day.
        total_mand = 19
        if mand_work_count > total_mand:
            return False

        # Hard Constraint 2: Overtime period for every driver is 2 hours 30mins per day.
        total_overtime = 6
        if over_work_count > total_overtime:
            return False

        # Hard Constraint 3: Every work should be at least 1hr and does not exceed 4hr
        if (max_consecutive_work > 10) or (min_consecutive_work < 2):
            return False

        # Hard Constraint 4: No breaks at first hour
        if (arr[0] == 0) or (arr[1] == 0):
            return False

        if (break_count) != 6 or (max_consecutive_break > 2):
            return False

        return True

    def calc_fitness(self, ideal_break=8, ideal_overtime_break=2, overtime_idx=24):
        slots_matrix = np.stack(list(self.slots.values()), axis=0)

        # Soft Constraint 1: Number of drivers that goes on breaks together are preferabably the ideal value
        break_count_col = np.sum((slots_matrix == 0), axis=0)
        difference = np.abs(break_count_col - ideal_break)

        ignore_idx = [0, 1]

        penalty_score = np.zeros(
            len(difference),
        )

        for i, val in enumerate(difference):
            if i in ignore_idx:
                continue

            if val >= 3:
                penalty = 30
            elif 1 <= val <= 2:
                penalty = val * 10
            else:
                penalty = 0

            penalty_score[i] = penalty

        fitness1 = np.sum(penalty_score)

        # Soft Constraint 2: One hour break during overtime and 2 hour  is preferable
        overtime_break_count_row = np.sum((slots_matrix[:, overtime_idx:] == 0), axis=1)
        difference = np.abs(overtime_break_count_row - ideal_overtime_break)
        difference = [10 if val >= 1 else 0 for val in difference]

        fitness2 = np.sum(difference)

        fitness_score = fitness1 + fitness2

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

    print(s1.calc_fitness())

    # print("After mutation:\n")
    # s1.random_mutate(0.5)
    # s1.print_data()
