import torch


from chromosomesNAS201 import *

import torch
from chromosomesNAS201 import NAS_BENCH_201, chromosome  # Ensure these are correctly imported

class Population:
    def __init__(self, pop_size, num_edges, device=torch.device("cpu")):
        self._pop_size = pop_size
        self._num_edges = num_edges
        self._device = device
        # Initialize population
        self.population = [chromosome(self._num_edges, self._device, NAS_BENCH_201) for _ in range(pop_size)]

    def get_population_size(self):
        return len(self.population)

    def get_population(self):
        return self.population

    def print_population(self):
        for p in self.population:
            print(p.get_fitness())

    def pop_sort(self):
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)

    def random_pop(self):
        self.population = [chromosome(self._num_edges, self._device, NAS_BENCH_201) for _ in range(self._pop_size)]

    def pop_pop(self, indices_to_pop):
        for index in sorted(indices_to_pop, reverse=True):
            self.population.pop(index)

    def set_population(self, new_population):
        """Update the population with new architecture parameters."""
        for i, ind in enumerate(self.population):
            ind.arch_parameters = new_population[i]  # Ensure new_population[i] is a tensor

    def set_arch_parameters(self, new_arch_parameters):
        """Update architecture parameters for each individual in the population."""
        for individual, params in zip(self.population, new_arch_parameters):
            alphas_normal = params[0]
            alphas_reduce = params[1]
            individual.set_arch_parameters(alphas_normal, alphas_reduce)

    def get_arch_parameters(self):
        arch_params = [individual.get_arch_parameters() for individual in self.population]
        for idx, params in enumerate(arch_params):
            print(f"[DEBUG] get_arch_parameters - Param {idx}: shape = {params.shape}")
        return arch_params

    def get_population_tensor(self):
        architectures = [individual.arch_parameters[0].flatten().to(self._device) for individual in self.population]
        return torch.stack(architectures)

    def set_population_from_tensor(self, population_tensor):
      num_individuals, _, num_parameters = population_tensor.shape
      individual_size = self._num_edges * 5  # Assuming target shape [6,5]

      for i, individual in enumerate(self.population):
        # Ensure we're slicing the correct dimensions, depth should match number of splits
        individual_vector = population_tensor[i].view(-1)[:individual_size]
        individual.arch_parameters[0] = individual_vector.reshape(self._num_edges, -1).to(self._device)


