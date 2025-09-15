import time
import numpy as np


# Garter Snake Optimization Algorithm(GSO)
def GSOA(population, fobj, VRmin, VRmax, Max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    c1 = 1.0  # Acceleration coefficient
    c2 = 1.0  # Acceleration coefficient

    best_solution = np.zeros((dim, 1))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        for i in range(pop_size):
            snake = population[i]
            fitness = fobj(snake)

            if fitness < best_fitness:
                best_solution = snake
                best_fitness = fitness

            # Update snake's position
            direction = np.random.rand(dim)
            inertia = 0.5
            personal_attraction = c1 * np.random.rand(dim) * (best_solution - snake)
            global_attraction = c2 * np.random.rand(dim) * (best_solution - snake)
            velocity = inertia * direction + personal_attraction + global_attraction
            snake = snake + velocity

            # Ensure that the snake's position is within the search space
            snake = np.clip(snake, -10, 10)  # Example range: -10 to 10

            population[i] = snake
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_solution = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_solution, Convergence_curve, best_fitness, ct
