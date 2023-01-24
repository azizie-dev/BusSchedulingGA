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
            new_schedule[driver] = new_times

        self.slots.update(new_schedule)
        self.slots_matrix = np.stack(list(self.slots.values()), axis=0)

    def _generate_raw_times(self):
        new_times = np.ones(len(self.timeslots), dtype=int)
        random_indices = np.random.choice(
            np.arange(2, len(self.timeslots) - 1, 1), 6, replace=False
        )
        # random_indices = np.concatenate((random_indices, random_indices + 1), axis=0)
        new_times[random_indices] = 0

        return new_times

    def _calc_valid_times(self, arr, overtime_idx=23):

        penalty_score = 0
        # Hard Constraint 1: The mandatory working period for every driver is 8 hours per day.
        mand_work_count = np.sum(arr[:overtime_idx] == 1)
        total_mand = 19
        # if mand_work_count > total_mand:
        #     return False

        penalty_score += np.abs(total_mand - mand_work_count) * 100

        # Hard Constraint 2: Overtime period for every driver is 2 hours 30mins per day.
        total_overtime = 6
        over_work_count = np.sum(arr[overtime_idx:] == 1)

        # if over_work_count > total_overtime:
        #     return False
        penalty_score += np.abs(total_overtime - over_work_count) * 100

        consecutive_count = [
            len(list(group)) for key, group in groupby(arr) if key == 1
        ]

        max_consecutive_work = max(consecutive_count)
        min_consecutive_work = min(consecutive_count)

        # Hard Constraint 3: Every work should be at least 1hr and does not exceed 4hr
        if max_consecutive_work > 10:
            penalty_score += np.abs(max_consecutive_work - 10) * 100

        if min_consecutive_work < 2:
            penalty_score += np.abs(min_consecutive_work - 2) * 50

        # Hard Constraint 4: No breaks at first hour
        # if (arr[0] == 0) or (arr[1] == 0):
        #     penalty_score += 1000

        max_consecutive_break = min(
            [len(list(group)) for key, group in groupby(arr) if key == 0]
        )
        break_count = np.sum(arr == 0)

        if (break_count) > 6:
            penalty_score += np.abs(break_count - 6) * 20

        if max_consecutive_break > 2:
            penalty_score += np.abs(max_consecutive_break - 2) * 10

        return penalty_score

    def calc_fitness(self, ideal_break=8, ideal_overtime_break=2, overtime_idx=24):

        fitness0 = np.sum([self._calc_valid_times(time) for time in self.slots_matrix])

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

        fitness_score = fitness0 + fitness1 + fitness2

        return fitness_score

    def random_mutate(self, mutation_rate):
        rand = np.random.rand(self.slots_matrix.shape[0], self.slots_matrix.shape[1])
        change_mask = rand < mutation_rate
        self.slots_matrix[change_mask] = 1 - self.slots_matrix[change_mask]
        self.slots_matrix[:, :2] = 1

    def random_crossover(self, other_schedule, crossover_rate=0.8):

        # # Generate random crossover points
        # crossover_points = np.random.randint(0, self.slots_matrix.shape[1], size=points)

        # # Sort the crossover points
        # crossover_points.sort()

        # # Add column 0 and the last column
        # crossover_points = np.concatenate(
        #     ([0], crossover_points, [self.slots_matrix.shape[1] - 1])
        # )

        # # Create two empty arrays to store the child arrays
        # child1 = np.empty_like(self.slots_matrix)
        # child2 = np.empty_like(self.slots_matrix)

        # # Iterate over the columns of the array
        # for i in range(points + 1):
        #     if i % 2 == 0:
        #         child1[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ] = self.slots_matrix[:, crossover_points[i] : crossover_points[i + 1]]
        #         child2[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ] = other_schedule.slots_matrix[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ]
        #     else:
        #         child1[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ] = other_schedule.slots_matrix[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ]
        #         child2[
        #             :, crossover_points[i] : crossover_points[i + 1]
        #         ] = self.slots_matrix[:, crossover_points[i] : crossover_points[i + 1]]

        # Create two empty arrays to store the child arrays
        child1 = np.empty_like(self.slots_matrix)
        child2 = np.empty_like(self.slots_matrix)

        # Iterate over the elements of the array
        for i in range(self.slots_matrix.shape[0]):
            for j in range(self.slots_matrix.shape[1]):
                if np.random.rand() < crossover_rate:
                    child1[i, j] = other_schedule.slots_matrix[i, j]
                    child2[i, j] = self.slots_matrix[i, j]
                else:
                    child1[i, j] = self.slots_matrix[i, j]
                    child2[i, j] = other_schedule.slots_matrix[i, j]

        child1[:, :2] = 1
        child2[:, :2] = 1

        new_schedule1, new_schedule2 = Schedule(), Schedule()
        new_schedule1.slots_matrix = child1
        new_schedule2.slots_matrix = child2

        return new_schedule1, new_schedule2


class GeneticAlgorithm:
    def __init__(self):
        self.final_generation = 10

        self.fitness_vals = []
        self.population_size = 10
        self.populations: list[Schedule] = []

        self.crossover_rate = 0.8
        self.mutation_rate = 0.5

    def run(self):
        self._generate_population(self.population_size)

        for i in range(self.final_generation + 1):

            # Sort by fitness values
            self.populations.sort(key=lambda x: x.calc_fitness(), reverse=False)
            if ((i + 1) % 5) == 0:
                print(f"Best Schedule in Generation {i+1}\n")
                self.populations[0].print_data()
                print()

            self.fitness_vals.append(self.populations[0].calc_fitness())
            worst_chromosome = self.populations.pop()
            worst_chromosome_fitness = worst_chromosome.calc_fitness()

            attempts = 0
            # Mutate until better chromosome is achieved

            while True:
                self._select_chromosome()
                self._crossover_chromosome()
                self._mutate_chromosome()

                # Check if children is better than the worst chromosome
                if self.mating_pool[0].calc_fitness() <= worst_chromosome_fitness:
                    break

                if self.mating_pool[1].calc_fitness() <= worst_chromosome_fitness:
                    break

                attempts += 1

                # if attempts > 10000:
                #     break

            feasible_chromosome = sorted(
                [self.mating_pool[0], self.mating_pool[1], worst_chromosome],
                key=lambda x: x.calc_fitness(),
            )[0]
            self.populations.append(feasible_chromosome)

    def _generate_population(self, population_size=10):
        # Initialize population
        for _ in range(population_size):
            s = Schedule()
            s.generate_random_schedule()
            self.populations.append(s)

    def _select_chromosome(self):
        # Select 2 then the best one is selected, the selected are removed so that it will not get selected the second time
        self.mating_pool: list[Schedule] = []

        fitness_values = [x.calc_fitness() for x in self.populations]
        min_fitness = min(fitness_values)
        fitness_values = [min_fitness - i for i in fitness_values]
        weights = [i / sum(fitness_values) for i in fitness_values]

        # Select an individual
        selected_individuals = random.choices(self.populations, weights=weights, k=2)

        self.mating_pool.append(selected_individuals[0])
        self.mating_pool.append(selected_individuals[1])

    def _crossover_chromosome(self):
        # Crossover of the parents
        child1, child2 = self.mating_pool[0].random_crossover(
            self.mating_pool[1], self.crossover_rate
        )

        self.mating_pool[0] = child1
        self.mating_pool[1] = child2

    def _mutate_chromosome(self):
        # Mutation of the chromosome
        self.mating_pool[0].random_mutate(mutation_rate=self.mutation_rate)
        self.mating_pool[1].random_mutate(mutation_rate=self.mutation_rate)


if __name__ == "__main__":
    gen = GeneticAlgorithm()
    gen.run()
