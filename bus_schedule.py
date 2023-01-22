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
        self.slots_matrix: np.ndarray

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
        self.data_from_matrix()
        pprint.pprint(self.slots)
        print(f"\nFitness: {self.calc_fitness()}\n")

    def data_from_matrix(self):
        new_dict = {
            driver: time for driver, time in zip(self.drivers, self.slots_matrix)
        }
        self.slots.update(new_dict)

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
        self.slots_matrix = np.stack(list(self.slots.values()), axis=0)

    def _generate_raw_times(self):
        new_times = np.ones(len(self.timeslots), dtype=int)
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

        for time in self.slots_matrix:
            if not (self._is_valid_times(time)):
                return 100000

        # Soft Constraint 1: Number of drivers that goes on breaks together are preferabably the ideal value
        break_count_col = np.sum((self.slots_matrix == 0), axis=0)
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
        overtime_break_count_row = np.sum(
            (self.slots_matrix[:, overtime_idx:] == 0), axis=1
        )
        difference = np.abs(overtime_break_count_row - ideal_overtime_break)
        difference = [10 if val >= 1 else 0 for val in difference]

        fitness2 = np.sum(difference)

        fitness_score = fitness1 + fitness2

        return fitness_score

    def random_mutate(self, mutation_rate=0.1):
        rand = np.random.rand(self.slots_matrix.shape[0], self.slots_matrix.shape[1])
        change_mask = rand < mutation_rate
        self.slots_matrix[change_mask] = 1 - self.slots_matrix[change_mask]

    def random_crossover(self, other_schedule, points=3):

        # Generate random crossover points
        crossover_points = np.random.randint(
            0, self.slots_matrix.shape[1], size=(self.slots_matrix.shape[0], points)
        )

        # Sort the crossover points
        crossover_points.sort(axis=-1)

        # Add column 0 and the last column
        crossover_points = np.concatenate(
            (
                np.zeros((self.slots_matrix.shape[0], 1), dtype=int),
                crossover_points,
                np.array(
                    [self.slots_matrix.shape[1] - 1] * self.slots_matrix.shape[0]
                ).reshape(self.slots_matrix.shape[0], 1),
            ),
            axis=1,
        )

        # Create two empty arrays to store the child arrays
        child1 = np.empty_like(self.slots_matrix)
        child2 = np.empty_like(self.slots_matrix)

        # Iterate over the rows of the array
        for i in range(self.slots_matrix.shape[0]):
            for j in range(points + 1):
                if j % 2 == 0:
                    child1[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ] = self.slots_matrix[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ]
                    child2[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ] = other_schedule.slots_matrix[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ]
                else:
                    child1[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ] = other_schedule.slots_matrix[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ]
                    child2[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ] = self.slots_matrix[
                        i, crossover_points[i, j] : crossover_points[i, j + 1]
                    ]

        new_schedule1, new_schedule2 = Schedule(), Schedule()
        new_schedule1.slots_matrix = child1
        new_schedule2.slots_matrix = child2

        return new_schedule1, new_schedule2


class GeneticAlgorithm:
    def __init__(self):
        self.final_generation = 1

        self.fitness_vals = []
        self.population_size = 10
        self.populations: list[Schedule] = []

    def run(self):
        self._generate_population(self.population_size)

        for i in range(self.final_generation + 1):

            # Sort by fitness values
            self.populations.sort(key=lambda x: x.calc_fitness(), reverse=False)
            self.fitness_vals.append(self.populations[0].calc_fitness())
            worst_chromosome = self.populations.pop()

            self._select_chromosome()
            self._crossover_chromosome(crossover_point=3)
            self._mutate_chromosome(mutation_rate=0.1)

            # Check if children is better than the worst chromosome
            while (
                self.mating_pool[0].calc_fitness() >= worst_chromosome.calc_fitness()
                and self.mating_pool[1].calc_fitness()
                >= worst_chromosome.calc_fitness()
            ):
                # Mutate until better chromosome is achieved
                self._select_chromosome()
                self._crossover_chromosome(crossover_point=3)
                self._mutate_chromosome(mutation_rate=0.5)

            feasible_chromosome = sorted(
                [self.mating_pool[0], self.mating_pool[1], worst_chromosome],
                key=lambda x: x.calc_fitness(),
            )[0]
            self.populations.append(feasible_chromosome)

            if ((i + 1) % 10) == 0:
                print(f"Best Schedule in Generation {i+1}\n")
                self.populations[0].print_data()
                print()

    def _generate_population(self, population_size=10):
        # Initialize population
        for _ in range(population_size):
            s = Schedule()
            s.generate_random_schedule()
            self.populations.append(s)

    def _select_chromosome(self):
        # Select 2 then the best one is selected, the selected are removed so that it will not get selected the second time
        self.mating_pool: list[Schedule] = []
        selected = sorted(
            random.sample(self.populations, 4), key=lambda x: x.calc_fitness()
        )
        self.mating_pool.append(selected[0])
        self.mating_pool.append(selected[1])

    def _crossover_chromosome(self, crossover_point=3):
        # Crossover of the parents
        child1, child2 = self.mating_pool[0].random_crossover(
            self.mating_pool[1], crossover_point
        )

        self.mating_pool[0] = child1
        self.mating_pool[1] = child2

    def _mutate_chromosome(self, mutation_rate=0.1):
        # Mutation of the chromosome
        self.mating_pool[0].random_mutate(mutation_rate=mutation_rate)
        self.mating_pool[1].random_mutate(mutation_rate=mutation_rate)


if __name__ == "__main__":
    gen = GeneticAlgorithm()
    GeneticAlgorithm().run()
