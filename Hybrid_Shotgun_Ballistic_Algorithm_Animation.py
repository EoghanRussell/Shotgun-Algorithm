import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


""" 2016 Dissertation by Eoghan Russell and Stephen Dillon - Hybrid Shotgun ballistic algorithm
    This code is a 3D animation of 3 functions, Rosenbrock, Rastrigin, Ackley. Its aim is to give the user a graphical
    feel of how the algorithm works.
    When viewing the results in "Final Results" mode this will give a graphical representation of all function
    evalutions made throughout the algorithm. The user will see how momentum increases particle moves when fitness
    improves and decreases as fitness dis-improves
    When viewing the results in "Animation" mode this will show the algorithms progression over time. First all randomly searched particles
    are displayed. This is followed by every 10th iteration of the GA so the user can see it improve over time.
    This is then followed by search using momentum.
    On both algorithms gbest is represented by a large red ball (when the run is complete)
"""
__author__ = 'Eoghan & Stephen'

""" ************************        Functions to Optimise       ************************ """

"""************************ Rastrigin's fitness: ************************"""
def rastrigin(X):
    """ Rastrigin's fitness: multimodal, symmetric, separable   """
    x = X[0]
    y = X[1]
    fitness = (((x ** 2) - 10 * math.cos(2 * math.pi * x)) + ((y ** 2) - 10 * math.cos(2 * math.pi * y))) / 2
    return fitness

    """************************ Rosenbrock's fitness: ************************"""
def rosenbrock(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    fitness = a*a + b*b*100
    return fitness

    """************************ Ackley's fitness: ************************"""
def ackley(x):
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    fitness = -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    return fitness

""" ************************        Shrink search space around a genetic position       ************************ """
""" ***************   Crossover/mutation of current g_best and position in genetic population   **************** """
def genetic_restart(lbounds,ubounds, g_best, dim, potential_positions_array):
    size = len(potential_positions_array)
    position_1 = random.randint(0, size - 1)
    position_2 = random.randint(0, size - 1)
    chromosome_1 = potential_positions_array[position_1]
    chromosome_2 = potential_positions_array[position_2]
    new_chromosome_1 = []
    new_chromosome_2 = []
    y = 0
    ### Crossover
    crossover_point = random.randint(1, dim - 1)
    while y < dim:
        if y < crossover_point:
            new_chromosome_1.append(chromosome_1[y])
            new_chromosome_2.append(chromosome_2[y])
        else:
            temp = chromosome_1[y]
            new_chromosome_1.append(chromosome_2[y])
            new_chromosome_2.append(temp)
        ### Mutation
        if random.uniform(0, 1) > 0.99:
            new_chromosome_1[y] = random.uniform(lbounds[y], ubounds[y])
            new_chromosome_2[y] = random.uniform(lbounds[y], ubounds[y])
        y += 1
    if random.uniform(0,1) < 0.5:
        return new_chromosome_1
    else:
        return new_chromosome_2

""" ***************         Roulette wheel for chromosome selection (Rank)           ***************     """
def roulette (fitnessScores):
    cumalative_fitness = 0.0
    r = random.uniform(0, 1)
    for i in range(len(fitnessScores)):
        cumalative_fitness += fitnessScores[i] # add ranks until value rached
        if cumalative_fitness > r: # when value > r, return index of the chromosome to use
            return i

""" ***************         Genetic Algorithm           ***************     """
def genetic_Algorithm(function, genetic_fitness, genetic_position, global_best_fitness, g_best, dim, budget,lbounds,ubounds, x_plot, gbest_array, count, local_position_array, fitness_array):
    generation = 1
    while(generation <= 100):
        next_generation_position = []
        next_generation_fitness = []
        population = len(genetic_position)
        count1 = 0
        ### Get best 10% of the population (this will be automatically copied over to the next generation)
        while count1 < population * 0.1:
            min_value = min(genetic_fitness)
            min_index = genetic_fitness.index(min_value)
            next_generation_position.append(genetic_position[min_index])
            next_generation_fitness.append(genetic_fitness[min_index])
            genetic_position.remove(genetic_position[min_index])
            genetic_fitness.remove(genetic_fitness[min_index])
            count1 += 1
        ### add elite positions and fitness back into the population
        genetic_position += next_generation_position
        genetic_fitness += next_generation_fitness
        ### Order arrays so Rank selection can be used
        count1 = 0
        ordered_genetic_position = []
        fitness_rank = []
        additive = 0
        q = 1
        # additive will be used as a divisor to calculate rank
        while q < population:
            additive += q
            q += 1
        # create ordered arrays
        while count1 < population:
            min_value = min(genetic_fitness)
            min_index = genetic_fitness.index(min_value)
            ordered_genetic_position.append(genetic_position[min_index])
            fitness_rank.append((float(population) - float(count1))/float(additive))
            genetic_position.remove(genetic_position[min_index])
            genetic_fitness.remove(genetic_fitness[min_index])
            count1 += 1
        #### Generate the next generation
        count_chromosomes = len(next_generation_position)
        index1 = 0
        index2 = 0
        while count_chromosomes < population:
            ### Select Chromosomes to perform crossover/mutation
            same_index = True
            while same_index == True:
                index1 = roulette(fitness_rank)
                index2 = roulette(fitness_rank)
                if index1 != index2:
                    same_index = False
            chromosome_1 = ordered_genetic_position[index1]
            chromosome_2 = ordered_genetic_position[index2]
            new_chromosome_1 = []
            new_chromosome_2 = []
            y = 0
            ### Crossover
            crossover_point = random.randint(1, dim - 1)
            while y < dim:
                if y < crossover_point:
                    new_chromosome_1.append(chromosome_1[y])
                    new_chromosome_2.append(chromosome_2[y])
                else:
                    temp = chromosome_1[y]
                    new_chromosome_1.append(chromosome_2[y])
                    new_chromosome_2.append(temp)
                ### Mutation
                if random.uniform(0, 1) > 0.9:
                    new_chromosome_1[y] = random.uniform(lbounds[y], ubounds[y])
                    new_chromosome_2[y] = random.uniform(lbounds[y], ubounds[y])
                y += 1
            ### Add chromosomes to the next generation
            next_generation_position.append(new_chromosome_1)
            fitness_value1 = function(list(new_chromosome_1))
            next_generation_fitness.append(fitness_value1)
            local_position_array.append(new_chromosome_1)
            fitness_array.append(fitness_value1)
            x_plot.append(count)
            gbest_array.append(g_best)
            count += 1
            budget -= 1
            if count_chromosomes != population - 1:
                next_generation_position.append(new_chromosome_2)
                fitness_value2 = function(list(new_chromosome_2))
                next_generation_fitness.append(fitness_value2)

                local_position_array.append(new_chromosome_2)
                fitness_array.append(fitness_value2)
                x_plot.append(count)
                gbest_array.append(g_best)
                count += 1
                budget -= 1
            count_chromosomes += 2
        ### Make current generation = next generation (for next loop of the genetic algorithm)
        genetic_fitness = next_generation_fitness
        genetic_position = next_generation_position
        generation += 1
    ### Update g_best if applicable
    fittest_chromosome = min(genetic_fitness)
    fittest_index = genetic_fitness.index(fittest_chromosome)
    if fittest_chromosome < global_best_fitness:
        global_best_fitness = fittest_chromosome
        g_best = genetic_position[fittest_index]
    return g_best, global_best_fitness, genetic_position, budget, x_plot, gbest_array, count, local_position_array, fitness_array

""" ************************************************************************************ """
""" ************************        Create new particle         ************************ """
def new_individual(lbounds, ubounds, dim):
    i = 0
    position = []
    while i < dim:
        position.append(random.uniform(lbounds[i], ubounds[i])) # Randomly create a point within the search space
        i += 1
    return tuple(position)

""" ************************        Move particle               ************************ """
def move_particle(local_position, momentum_magnitude, lbounds, ubounds, momentum_direction):
    new_position = []
    i = 0
    dimensions = len(local_position)
    while i < dimensions:
        # new_position is a function of the current position plus a directional proportion of the search space
        new_position.append(local_position[i] + ((momentum_magnitude * (ubounds[i] - lbounds[i])) * momentum_direction[i]))
        # If the new position is outside the search space bring it back inside
        if new_position[i] < lbounds[i]:
            new_position[i] = lbounds[i]
        elif new_position[i] > ubounds[i]:
            new_position[i] = ubounds[i]
        i += 1
    return tuple(new_position)

""" ************************        Move particle PSO           ************************ """
def move_particle_pso(local_position, momentum_magnitude, lbounds, ubounds, momentum_direction, g_best, progress):
    new_position = []
    i = 0
    dimensions = len(local_position)
    while i < dimensions:
        new_position.append(local_position[i] + ((1-progress) * ((momentum_magnitude * (ubounds[i] - lbounds[i])) * momentum_direction[i])) \
                          + (progress * momentum_magnitude * g_best[i]))
        # If the new position is outside the search space bring it back inside
        if new_position[i] < lbounds[i]:
            new_position[i] = lbounds[i]
        elif new_position[i] > ubounds[i]:
            new_position[i] = ubounds[i]
        i += 1
    return tuple(new_position)

""" ************************        Move particle Balance           ************************ """
def move_particle_balance(local_position, momentum_magnitude, lbounds, ubounds, progress):
    new_position = []
    power = 0.05
    movement_function = min(-np.log(progress) * ((1 - progress) ** power), (1 - progress) ** power)
    i = 0
    dimensions = len(local_position)
    while i < dimensions:
        new_position.append(local_position[i] + (momentum_magnitude * movement_function))
        if new_position[i] < lbounds[i]:
            new_position[i] = lbounds[i]
        elif new_position[i] > ubounds[i]:
            new_position[i] = ubounds[i]
        i += 1
    return tuple(new_position)

""" ************************     Create new particle (Shrink Searchspace) ************************ """
def new_individual_shrink_ws(lbounds, ubounds, g_best, dim, progress):
    ind_shk_ws = []
    i = 0
    gamma = 0.01
    while i < dim:
        # reduction value reduces as progress increases to fine tune around gbest as the algorithm progresses
        reduction = ((1 - progress)**2) * (abs(ubounds[i] - lbounds[i])) * gamma
        # Reduce the search space about the global best
        ind_shk_ws.append(random.uniform(max(g_best[i] - reduction, lbounds[i]),
                                  min(g_best[i] + reduction, ubounds[i])))
        i += 1
    return tuple(ind_shk_ws)

""" ************************    Create new particle (Simulated annealing) ************************ """
def new_individual_SA(lbounds, ubounds, g_best, dim, total_budget, budget):
    ind_shk_ws = []
    i = 0
    alpha = 0.99
    while i < dim:
        # restriction is decreasing
        restriction = alpha**(total_budget-budget)
        # Reduce the search space about the global best
        ind_shk_ws.append(random.uniform(max(g_best[i] - restriction, lbounds[i]),
                                  min(g_best[i] + restriction, ubounds[i])))
        i += 1
    return tuple(ind_shk_ws)

""" ************************        Calculate momentum_magnitude          ************************ """
def calc_momentum(local_fitness, previous_fitness, previous_momentum):
    allowed_range = 0.05
    epsilon = 0.000001
    if abs(previous_fitness - local_fitness) < epsilon:
        percent_fit_diff = 0.01
    else:
        percent_fit_diff = abs((previous_fitness - local_fitness)/previous_fitness)
    if local_fitness <= previous_fitness:
        # Increase momentum_magnitude
        momentum_magnitude = min(((1 + percent_fit_diff) * previous_momentum),
                1.1 * previous_momentum, allowed_range)
    else:
        # Otherwise decrease momentum_magnitude
        momentum_magnitude = max(previous_momentum / 2.0, 0.0)
    return momentum_magnitude

""" ************************        Main function        ************************ """
def random_search(function, lbounds, ubounds, budget):
    """ ***************     Define globals for matplotlib       *************** """
    global count
    count = 1
    global local_position_array
    local_position_array = []
    global fitness_array
    fitness_array = []
    global x_plot
    x_plot = []
    global momentum_array
    momentum_array = []
    global gbest_array
    gbest_array = []
    global global_best_fitness_array
    global_best_fitness_array = []
    global g_best
    g_best = []
    global global_best_fitness
    epsilon = 0.000001
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)
    restarts = 0
    wrong_way_steps = 0
    total_budget = budget
    potential_positions_array = []
    momentum_magnitude = random.uniform(0.01, 0.05)

    """ ***************     Define momentum_direction        *************** """
    i = 0
    momentum_direction = []
    while i < dim:
        momentum_direction.append(random.uniform(-1, 1))
        i += 1

    """ ***************     Use multiple starting positions to define an initial swarm  ***************
        ***************     Use best point as starting point, pbest, and gbest          ***************"""
    local_position = new_individual(list(lbounds), list(ubounds), dim)
    local_fitness = function(local_position)
    budget -= 1
    random_starts = 1
    """ ***************     Update arrays used in plots       ***************     """
    local_position_array.append(local_position)
    fitness_array.append(local_fitness)
    x_plot.append(count)
    momentum_array.append(abs(momentum_magnitude))
    gbest_array.append(g_best)
    global_best_fitness_array.append(local_fitness)
    count += 1
    """ ***************     Initial swarm                   ***************     """
    genetic_position = []
    genetic_fitness = []
    while random_starts < total_budget * 0.05:
        random_position = new_individual(list(lbounds), list(ubounds), dim)
        random_fitness = function(random_position)
        budget -= 1
        """ ***************     Update arrays used in plots ***************     """
        local_position_array.append(random_position)
        fitness_array.append(random_fitness)
        x_plot.append(count)
        momentum_array.append(abs(momentum_magnitude))
        gbest_array.append(g_best)
        global_best_fitness_array.append(random_fitness)
        count += 1
        if total_budget > 1000:
            if total_budget < 3500:
                population_size = 5
            else:
                population_size = total_budget * 0.0015
            if random_starts <= population_size and (len(genetic_position) <= 150):
                genetic_position.append(random_position)
                genetic_fitness.append(random_fitness)
            else:
                max_value = max(genetic_fitness)
                if(random_fitness < max_value):
                    max_index = genetic_fitness.index(max_value)
                    genetic_fitness.remove(genetic_fitness[max_index])
                    genetic_position.remove(genetic_position[max_index])
                    genetic_position.append(random_position)
                    genetic_fitness.append(random_fitness)

        """ ***************     Update local position       ***************     """
        if random_fitness < local_fitness:
            local_position = g_best = random_position
            local_fitness = global_best_fitness = random_fitness
        random_starts += 1

    """ ***************     gbest positions    ***************     """
    g_best = local_position
    global_best_fitness = local_fitness

    """ ***************     Run genetic Algorithm    ***************     """
    if len(g_best) > 1 and total_budget > 1000:
        g_best, global_best_fitness, potential_positions_array, budget, x_plot, gbest_array, count, local_position_array, fitness_array = genetic_Algorithm(function, genetic_fitness, genetic_position, global_best_fitness, g_best, dim, budget, lbounds, ubounds, x_plot, gbest_array, count, local_position_array, fitness_array)

    """ ***************     Main while loop begins here     ***************     """
    while budget > 0:
        budget -= 1
        progress = 1 - (float(budget)/float(total_budget))

        """ ***************     Get next position/fitness ***************     """
        if movement_method == 1:
            next_position = move_particle(list(local_position), momentum_magnitude, lbounds, ubounds, momentum_direction)
        elif movement_method == 2:
            next_position = move_particle_pso(list(local_position), momentum_magnitude, lbounds,
                                              ubounds, momentum_direction, g_best, progress)
        elif movement_method == 3:
            next_position = move_particle_balance(list(local_position), momentum_magnitude, lbounds, ubounds, momentum_direction, g_best, progress)
        next_fitness = function(next_position)

        """ ***************     Update arrays used in plots       ***************     """
        local_position_array.append(local_position)
        fitness_array.append(local_fitness)
        x_plot.append(count)
        momentum_array.append(abs(momentum_magnitude))
        gbest_array.append(g_best)
        global_best_fitness_array.append(local_fitness)
        count += 1

        """ ***************     If Ubound or lbound found change the direction of momentum  ***************     """
        """ ***************     Done to ensure 1 dimension doesn't get stuck at an extremum ***************     """
        i = 0
        while i < dim:
            if (abs(next_position[i] - lbounds[i]) < epsilon) or (abs(next_position[i] - ubounds[i]) < epsilon):
                momentum_direction[i] = -momentum_direction[i]
            i += 1

        """ ***************     Check fitness against gbest           ***************     """
        if next_fitness < global_best_fitness:
            global_best_fitness = next_fitness
            g_best = next_position

        """ ***************     Calculating momentum        ***************     """
        if next_fitness < local_fitness:
            if wrong_way_steps > 0:
                wrong_way_steps -= 1
            # increase momentum
            momentum_magnitude = calc_momentum(next_fitness, local_fitness, momentum_magnitude)
            # move to local position
            local_position = next_position
            # set new local fitness
            local_fitness = next_fitness
        else:
            wrong_way_steps += 1
            momentum_criteria = 0.001
            if (abs(momentum_magnitude) > (1 - progress) * momentum_criteria) and (wrong_way_steps < 8):
                # decrease momentum
                momentum_magnitude = calc_momentum(next_fitness, local_fitness, momentum_magnitude)
                local_position = next_position
                local_fitness = next_fitness
            elif budget > 0:
                # we now restart from a random position
                wrong_way_steps = 0
                momentum_magnitude = random.uniform(0.01, 0.05)
                momentum_direction = []
                i = 0
                while i < dim:
                    momentum_direction.append(random.uniform(-1, 1))
                    i += 1
                """ ***************     If we are 50% percent through the algorithm     ***************     """
                """ ***************     restrict search space with 90% probability      ***************     """
                if progress > 0.5:
                    i= 0
                    while i < dim:
                        if random.uniform(0,1) < 0.3:
                            momentum_direction[i] = 0
                        i += 1
                shrink_to_g_best = 0.5
                if progress >= shrink_to_g_best:
                    if shrink_method == 1:
                        restart_position = local_position = new_individual_shrink_ws(list(lbounds), list(ubounds), g_best, dim, progress)
                    elif shrink_method == 2:
                        restart_position = local_position = new_individual_SA(list(lbounds), list(ubounds), g_best,
                                                                    dim, total_budget, budget, alpha)
                    restart_fitness = local_fitness = function(local_position)
                    budget -= 1
                    local_position_array.append(restart_position)
                    if restart_fitness < global_best_fitness:
                        global_best_fitness = restart_fitness
                        g_best = restart_position
                    random_starts = 1
                    fitness_array.append(restart_fitness)
                    x_plot.append(count)
                    momentum_array.append(abs(momentum_magnitude))
                    gbest_array.append(g_best)
                    global_best_fitness_array.append(global_best_fitness)
                    count += 1
                    restarts += 1
                    while random_starts < total_budget * 0.0005 and budget > 0:
                        if shrink_method == 1:
                            restart_position = new_individual_shrink_ws(list(lbounds), list(ubounds), g_best, dim, progress)
                        elif shrink_method == 2:
                            restart_position = local_position = new_individual_SA(list(lbounds), list(ubounds), g_best,
                                                                    dim, total_budget, budget, alpha)
                        restart_fitness = function(restart_position)
                        budget -= 1
                        restarts += 1
                        if restart_fitness < local_fitness:
                            local_position  = restart_position
                            local_fitness = restart_fitness
                        if restart_fitness < global_best_fitness:
                            g_best = local_position
                            global_best_fitness = local_fitness
                        random_starts += 1
                    local_position_array.append(restart_position)
                    fitness_array.append(restart_fitness)
                    x_plot.append(count)
                    momentum_array.append(abs(momentum_magnitude))
                    count += 1
                else:
                    restart_position = local_position = genetic_restart(list(lbounds), list(ubounds), g_best, dim, potential_positions_array)
                    restart_fitness = local_fitness = particle_best_fitness = function(restart_position)
                    """ ***************     Update arrays used in plots       ***************     """
                    local_position_array.append(restart_position)
                    fitness_array.append(restart_fitness)
                    x_plot.append(count)
                    momentum_array.append(abs(momentum_magnitude))
                    gbest_array.append(g_best)
                    global_best_fitness_array.append(restart_fitness)
                    count += 1
                    budget -= 1
                    if restart_fitness < global_best_fitness:
                        global_best_fitness = restart_fitness
                        g_best = local_position
                    restarts += 1
                    random_starts = 1
                    while random_starts < total_budget * 0.0005 and budget > 0:
                        if random.uniform(0, 1) < 0.5 and len(g_best) > 1:
                            if len(potential_positions_array) == 0:
                                restart_position = new_individual(list(lbounds), list(ubounds), dim)
                            else:
                                if restart_method == 1:
                                    restart_position = new_individual(list(lbounds), list(ubounds), dim)
                                else:
                                    restart_position = genetic_restart(list(lbounds), list(ubounds), g_best, dim, potential_positions_array)
                        else:
                            restart_position = new_individual(list(lbounds), list(ubounds), dim)
                        restart_fitness = function(restart_position)
                        budget -= 1
                        restarts += 1
                        if restart_fitness < local_fitness:
                            local_position  =restart_position
                            local_fitness = restart_fitness
                            if restart_fitness < global_best_fitness:
                                global_best_fitness = restart_fitness
                                g_best = restart_position
                        random_starts += 1
                    local_position_array.append(restart_position)
                    fitness_array.append(restart_fitness)
                    x_plot.append(count)
                    momentum_array.append(abs(momentum_magnitude))
                    count += 1
    return global_best_fitness

""" ***************     Settable parameters     *************** """
#Create new particle (Shrink searchspace)
global gamma
gamma = 0.01
#new_individual_SA
global alpha
alpha = 0.99
#calc_momentum
global allowed_range
allowed_range = 0.05

""" ***************     User interface     ***************     """
global movement_method
global shrink_method
print "Please pick a function to optimise:"
print "1: Rosenbrock function"
print "2: Rastrigin function"
print "3: Ackyey function"
input_function = raw_input()
if (input_function <> "1") and (input_function <> "2")and (input_function <> "3"):
    print "Please select 1 or 2 or 3"
else:
    print "Do you want to move the particle using:"
    print "1: Standard method"
    print "2: Particle swarm optimisation"
    print "3: Balance method"
    movement_method = int(raw_input())
    print "Do you want to shrink the workspace using:"
    print "1: Standard method"
    print "2: Simulated annealing"
    shrink_method = int(raw_input())
    print "Should restarts be at random or from Genetic Population:"
    print "1: At Random"
    print "2: Genetic Population"
    restart_method = int(raw_input())
    print "Please input required budget (budget <= 1200):"
    budget = int(raw_input())
    if budget < 1200:
        budget = 1200
    print "Do you want to show final results or animate points?"
    print "1: Final Results"
    print "2: Animation"
    print "Note: Animation may take some time to run for high budgets"
    run_type = raw_input()
    print "Would you like to see graphs top down or side elevation? ?"
    print "1: Top down"
    print "2: Side elevation"
    elevation = raw_input()
    """ ***************     Graph the Rosenbrock function     *************** """
    if input_function == "1":
        random_search(rosenbrock, [-2, -1], [2, 3], budget)
        #************************************
        x_axis = [item[0] for item in local_position_array]
        y_axis = [item[1] for item in local_position_array]
        # Plot Rosenbrock surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if elevation=="1":
            ax.view_init(90,0)
        X_Rosen = np.arange(-2, 2, 0.05)
        Y_Rosen = np.arange(-1, 3, 0.05)
        X_Rosen, Y_Rosen = np.meshgrid(X_Rosen, Y_Rosen)
        Z_Rosen = (1 - X_Rosen)**2 + 100*(Y_Rosen - X_Rosen**2)**2
        surf_Rosen = ax.plot_surface(X_Rosen, Y_Rosen, Z_Rosen, rstride=1, cstride=1,
            cmap=cm.jet, norm = LogNorm(), linewidth=0, antialiased=False, alpha = 0.3)
        i = 0
        x_random = []
        y_random = []
        fitness_random = []
        while(i < budget):
            while(i < budget *0.05):
                x_random.append(x_axis[i])
                y_random.append(y_axis[i])
                fitness_random.append(fitness_array[i])
                i += 1
            if(i == budget * 0.05):
                if run_type == "2":
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='blue', s=10)
                    plt.pause(0.05)
                    random_print.remove()
            iteration = 0
            while(i < ((budget * 0.05) + (100 * budget * 0.0015))) and (i < len(x_axis)):
                generation = 0
                x_random = []
                y_random = []
                fitness_random = []

                while (generation < budget * 0.0015):
                    x_random.append(x_axis[i])
                    y_random.append(y_axis[i])
                    fitness_random.append(fitness_array[i])
                    generation += 1
                    i += 1
                if run_type == "2" and iteration%10 == 0:
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='yellow', s=40)
                    plt.pause(0.05)
                    random_print.remove()
                iteration += 1
            if i < len(x_axis):
                ax.scatter(x_axis[i], y_axis[i], fitness_array[i], c='black', s=5)
            if run_type == "2":
                plt.pause(0.000005)
            i += 1
            plt.draw()
        ax.scatter(g_best[0], g_best[1], global_best_fitness, c='red', s=100)
        plt.show()

        """ ***************     Graph the Rastrigin function     ***************     """
    elif input_function == "2":
        random_search(rastrigin, [-5.12, -5.12], [5.12, 5.12], budget)
        #************************************
        x_axis = [item[0] for item in local_position_array]
        y_axis = [item[1] for item in local_position_array]
        #************************************
        # Plot Rastrigin surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if elevation=="1":
            ax.view_init(90,0)
        X_Rastrig = np.arange(-5.12, 5.12, 0.05)
        Y_Rastrig = np.arange(-5.12, 5.12, 0.05)
        X_Rastrig, Y_Rastrig = np.meshgrid(X_Rastrig, Y_Rastrig)
        Z_Rastrig = 20 + X_Rastrig**2 - 10 * np.cos(2 * np.pi * X_Rastrig) + Y_Rastrig**2 - 10 * np.cos(2
                                    * np.pi * Y_Rastrig)
        surf_Rastrig = ax.plot_surface(X_Rastrig, Y_Rastrig, Z_Rastrig, rstride=1, cstride=1,
            cmap=cm.jet, norm = LogNorm(), linewidth=0, antialiased=False, alpha = 0.3)
        i = 0
        x_random = []
        y_random = []
        fitness_random = []
        while(i < budget):
            while(i < budget *0.05):
                x_random.append(x_axis[i])
                y_random.append(y_axis[i])
                fitness_random.append(fitness_array[i])
                i += 1
            if(i == budget * 0.05):
                if run_type == "2":
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='blue', s=10)
                    plt.pause(0.05)
                    random_print.remove()
            iteration = 0
            while(i < ((budget * 0.05) + (100 * budget * 0.0015))) and (i < len(x_axis)):
                generation = 0
                x_random = []
                y_random = []
                fitness_random = []

                while (generation < budget * 0.0015):
                    x_random.append(x_axis[i])
                    y_random.append(y_axis[i])
                    fitness_random.append(fitness_array[i])
                    generation += 1
                    i += 1
                if run_type == "2" and iteration%10 == 0:
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='yellow', s=40)
                    plt.pause(0.05)
                    random_print.remove()
                iteration += 1
            if i < len(x_axis):
                ax.scatter(x_axis[i], y_axis[i], fitness_array[i], c='black', s=5)
            if run_type == "2":
                plt.pause(0.000005)
            i += 1
            plt.draw()
        ax.scatter(g_best[0], g_best[1], global_best_fitness, c='red', s=100)
        plt.show()
        """ ***************     Graph the Ackley function     ***************     """
    elif input_function == "3":
        random_search(ackley, [-40, -40], [40, 40], budget)
        #************************************
        x_axis = [item[0] for item in local_position_array]
        y_axis = [item[1] for item in local_position_array]
        #************************************
        # Plot Ackley surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if elevation=="1":
            ax.view_init(90,0)
        X_Ackley = np.arange(-40, 40, 0.5)
        Y_Ackley = np.arange(-40, 40, 0.5)
        X_Ackley, Y_Ackley = np.meshgrid(X_Ackley, Y_Ackley)
        Z_Ackley = -20. * np.exp(-0.2 * np.sqrt(0.5 * (X_Ackley ** 2 + Y_Ackley ** 2))) - np.exp(0.5 * (np.cos(2. * np.pi * X_Ackley) + np.cos(2. * np.pi * Y_Ackley))) + 20. + np.e
        surf_Ackley = ax.plot_surface(X_Ackley, Y_Ackley, Z_Ackley, rstride=1, cstride=1,
                        cmap=cm.jet, norm = LogNorm(), linewidth=0, antialiased=False, alpha = 0.3)
        i = 0
        x_random = []
        y_random = []
        fitness_random = []
        while(i < budget):
            while(i < budget *0.05):
                x_random.append(x_axis[i])
                y_random.append(y_axis[i])
                fitness_random.append(fitness_array[i])
                i += 1
            if(i == budget * 0.05):
                if run_type == "2":
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='blue', s=10)
                    plt.pause(0.05)
                    random_print.remove()
            iteration = 0
            while(i < ((budget * 0.05) + (100 * budget * 0.0015))) and (i < len(x_axis)):
                generation = 0
                x_random = []
                y_random = []
                fitness_random = []
                while (generation < budget * 0.0015):
                    x_random.append(x_axis[i])
                    y_random.append(y_axis[i])
                    fitness_random.append(fitness_array[i])
                    generation += 1
                    i += 1
                if run_type == "2" and iteration%10 == 0:
                    random_print = ax.scatter(x_random, y_random, fitness_random, c='yellow', s=40)
                    plt.pause(0.05)
                    random_print.remove()
                iteration += 1
            if i < len(x_axis):
                ax.scatter(x_axis[i], y_axis[i], fitness_array[i], c='black', s=5)
            if run_type == "2":
                plt.pause(0.000005)
            i += 1
            plt.draw()
        ax.scatter(g_best[0], g_best[1], global_best_fitness, c='red', s=100)
        plt.show()
