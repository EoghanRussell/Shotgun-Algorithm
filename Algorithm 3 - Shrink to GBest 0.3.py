#!/usr/bin/env python
"""Use case for the COCO experimentation module `cocoex` which can be used as
template.

Usage from a system shell::

    python example_experiment.py 3 1 20

runs the first of 20 batches with maximal budget
of 3 * dimension f-evaluations.

Usage from a python shell::

    # >>> import example_experiment as ee
    # >>> ee.main(3, 1, 1)  # doctest: +ELLIPSIS
    Benchmarking solver...

does the same but runs the "first" of one single batch.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
import random
from cocoex import Suite, Observer, log_level
verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass


def print_flush(*args):
    """print without newline and flush"""
    print(*args, end="")
    sys.stdout.flush()


def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


class ShortInfo(object):
    """print minimal info during benchmarking.

    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.

    Example output:

        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0
    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()
    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs
    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
        return s
    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
            self.f_current = f
        # print_flush(res)
        return res
    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")
    @staticmethod
    def short_time_stap():
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

# ===============================================
# prepare (the most basic example solver)
# ===============================================


def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if
    `problem_index + current_batch - 1` modulo `number_of_batches`
    equals to zero.
    """
    addressed_problems = []
    short_info = ShortInfo()
    for problem_index, problem in enumerate(suite):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension, max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations, runs)
        problem.free()
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    return addressed_problems

# ******************** Shotgun Algorithm ********************
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
def genetic_Algorithm(function, genetic_fitness, genetic_position, global_best_fitness, g_best, dim, budget,lbounds,ubounds):
    generation = 1
    while(generation <= 100):
        if generation == 2:
            generation = 2
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
                if random.uniform(0, 1) > 0.99:
                    new_chromosome_1[y] = random.uniform(lbounds[y], ubounds[y])
                    new_chromosome_2[y] = random.uniform(lbounds[y], ubounds[y])
                y += 1
            ### Add chromosomes to the next generation
            next_generation_position.append(new_chromosome_1)
            fitness_value1 = function(list(new_chromosome_1))
            next_generation_fitness.append(fitness_value1)
            budget -= 1
            if count_chromosomes != population - 1:
                next_generation_position.append(new_chromosome_2)
                fitness_value2 = function(list(new_chromosome_2))
                next_generation_fitness.append(fitness_value2)
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
    return g_best, global_best_fitness, genetic_position, budget

# ******************** Create new particle ********************
def new_individual(lbounds, ubounds, dim):
    i = 0
    position = []
    while i < dim:
        position.append(random.uniform(lbounds[i], ubounds[i]))
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
        # new_position is a function of the current position plus a directional proportion of the search space
        # plus a proportion of gbest
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

""" ************************     Create new particle (Shrink searchspace) ************************ """
def new_individual_shrink_ws(lbounds, ubounds, g_best, dim, progress):
    ind_shk_ws = []
    i = 0
    gamma = 0.0001
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
        # reduction is decreasing
        reduction = alpha**(total_budget-budget)
        # Reduce the search space about the global best
        ind_shk_ws.append(random.uniform(max(g_best[i] - reduction, lbounds[i]),
                                  min(g_best[i] + reduction, ubounds[i])))
        i += 1
    return tuple(ind_shk_ws)

""" ************************        Calculate momentum magnitude          ************************ """
def calc_momentum(local_fitness, previous_fitness, previous_momentum):
    allowed_range = 0.05
    epsilon = 0.000001
    if abs(previous_fitness - local_fitness) < epsilon:
        percent_fit_diff = 0.01
    else:
        percent_fit_diff = abs((previous_fitness - local_fitness)/previous_fitness)
    if local_fitness <= previous_fitness:
        # Increase momentum
        momentum_magnitude = min(((1 + percent_fit_diff) * previous_momentum),
                1.1 * previous_momentum, allowed_range)
    else:
        # Otherwise decrease momentum
        momentum_magnitude = previous_momentum / 2.0
    return momentum_magnitude

""" ************************        Main function           ************************ """
def random_search(function, lbounds, ubounds, budget):
    epsilon = 0.000001
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)
    restarts = 0
    wrong_way_steps = 0
    total_budget = budget
    potential_positions_array = []
    momentum_magnitude = random.uniform(0.01, 0.05)

    """ ***************     Define directed Momentum        *************** """
    i = 0
    momentum_direction = []
    while i < dim:
        momentum_direction.append(random.uniform(-1, 1))
        i += 1

    """ ***************     Use multiple starting positions to define an initial swarm  ***************
        ***************     Use best point as starting point, pbest, and gbest          ***************"""
    local_position = g_best = new_individual(list(lbounds), list(ubounds), dim)
    global_best_fitness = function(list(local_position))
    budget -= 1
    random_starts = 1
    """ ***************     Initial swarm                   ***************     """
    genetic_position = []
    genetic_fitness = []
    while random_starts < total_budget * 0.05:
        random_position = new_individual(list(lbounds), list(ubounds), dim)
        random_fitness = function(random_position)
        budget -= 1
        #### Setup Genetic algorithm with population of population_size % of the population or max_population
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
        if random_fitness < global_best_fitness:
            g_best = random_position
            global_best_fitness = random_fitness
        random_starts += 1

    """ ***************     Update gbest position   ***************     """
    local_position = g_best
    local_fitness = global_best_fitness

    """ ***************     Run genetic Algorithm    ***************     """
    if dim > 1 and total_budget > 1000:
         g_best, global_best_fitness, potential_positions_array, budget = genetic_Algorithm(function, genetic_fitness, genetic_position, global_best_fitness, g_best, dim, budget, lbounds, ubounds)

    """ ***************     Main while loop begins here     ***************     """
    while budget > 0:
        budget -= 1
        progress = 1 - (float(budget)/float(total_budget))

        """ ***************     Get next position/fitness ***************     """
        next_position = move_particle_pso(list(local_position), momentum_magnitude, lbounds, ubounds, momentum_direction, g_best, progress)
        next_fitness = function(list(next_position))

        """ ***************     If Ubound or lbound found change the direction of momentum_magnitude  ***************     """
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
        """ ***************     Calculating momentum_magnitude        ***************     """
        if next_fitness < local_fitness:
            if wrong_way_steps > 0:
                wrong_way_steps -= 1
            # increase momentum_magnitude
            momentum_magnitude = calc_momentum(next_fitness, local_fitness, momentum_magnitude)
            # move to local position
            local_position = next_position
            # set new local fitness
            local_fitness = next_fitness
        else:
            wrong_way_steps += 1
            momentum_criteria = 0.0001 # Adjustable Parameter
            # If momentum magnitude is greater than the minimum allowable value 
            # for this stage of the algorithm, then continue
            if (abs(momentum_magnitude) > (1 - progress) * momentum_criteria) and (wrong_way_steps < 8):
                # decrease momentum_magnitude
                momentum_magnitude = calc_momentum(next_fitness, local_fitness, momentum_magnitude)
                local_position = next_position
                local_fitness = next_fitness
            elif budget > 0:
                # Restart from a random position
                restarts += 1
                wrong_way_steps = 0
                momentum_magnitude = random.uniform(0.01, 0.05)
                momentum_direction = []
                i = 0
                while i < dim:
                    momentum_direction.append(random.uniform(-1, 1))
                    i += 1
                """ *** If shrink_to_g_best % through the algorithm shrink the search space around g_best *** """
                shrink_to_g_best = 0.3 # Adjustable Parameter
                if progress >= shrink_to_g_best:
                    local_position = new_individual_shrink_ws(list(lbounds), list(ubounds), g_best, dim, progress)
                    local_fitness = function(list(local_position))
                    budget -= 1
                    if local_fitness < global_best_fitness:
                        global_best_fitness = local_fitness
                        g_best = local_position
                    restarts += 1
                    random_starts = 1
                    while (random_starts < total_budget * 0.005) and (budget > 0):
                        restart_position = new_individual_shrink_ws(list(lbounds), list(ubounds), g_best , dim, progress)
                        restart_fitness = function(list(restart_position))
                        budget -= 1
                        if restart_fitness < local_fitness:
                            local_position = restart_position
                            local_fitness = restart_fitness
                        if restart_fitness < global_best_fitness:
                            g_best = local_position
                            global_best_fitness = local_fitness
                        random_starts += 1
                else:
                    local_position = new_individual(list(lbounds), list(ubounds), dim)
                    local_fitness = function(list(local_position))
                    budget -= 1
                    if local_fitness < global_best_fitness:
                        global_best_fitness = local_fitness
                        g_best = local_position
                    random_starts = 1
                    while (random_starts < total_budget * 0.005) and (budget > 0):
                        restart_position = new_individual(list(lbounds), list(ubounds), dim)
                        restart_fitness = function(list(restart_position))
                        budget -= 1
                        if restart_fitness < local_fitness:
                            local_position = restart_position
                            local_fitness = restart_fitness
                        if restart_fitness < global_best_fitness:
                            g_best = local_position
                            global_best_fitness = local_fitness
                        random_starts += 1

    return global_best_fitness


#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, max_runs=1e9):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    for restarts in range(int(max_runs)):
        remaining_evals = max_evals - fun.evaluations
        x0 = center + (restarts > 0) * 0.8 * range_ * (
                np.random.rand(fun.dimension) - 0.5)
        fun(x0)  # can be incommented, if this is done by the solver

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                   remaining_evals)
        elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
            if x0[0] == center[0]:
                sigma0 = 0.02
                restarts_ = 0
            else:
                x0 = "%f + %f * np.random.rand(%d)" % (
                        center[0], 0.8 * range_[0], fun.dimension)
                sigma0 = 0.2
                restarts_ = 6 * (observer_options.find('IPOP') >= 0)

            solver(fun, x0, sigma0 * range_[0], restarts=restarts_,
                   options=dict(scaling=range_/range_[0], maxfevals=remaining_evals,
                                termination_callback=lambda es: fun.final_target_hit,
                                verb_log=0, verb_disp=0, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_evals / fun.dimension,
                   iprint=-1)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        else:
            raise ValueError("no entry for solver %s" % str(solver.__name__))

        if fun.evaluations >= max_evals or fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        if fun.evaluations <= max_evals - remaining_evals:
            if max_evals - fun.evaluations > fun.dimension + 1:
                print("WARNING: %d evaluations remaining" %
                      remaining_evals)
            if fun.evaluations < max_evals - remaining_evals:
                raise RuntimeError("function evaluations decreased")
            break
    return restarts + 1

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
budget = 10000  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
SOLVER = random_search
#SOLVER = my_solver # fmin_slsqp # SOLVER = cma.fmin
#suite_name = "bbob-biobj"
suite_name = "bbob"
suite_instance = "year:2016"
suite_options = "dimensions: 2,3,5,10,20 "  # if 40 is not desired
observer_name = suite_name
observer_options = (
    ' result_folder: %s_on_%s_budget%04dxD '
                 % (SOLVER.__name__, suite_name, budget) +
    ' algorithm_name: %s ' % SOLVER.__name__ +
    ' algorithm_info: "A SIMPLE RANDOM SEARCH ALGORITHM" ')  # CHANGE THIS
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(budget=budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    `batch_loop(SOLVER, suite, observer, budget,...`.
    """
    observer = Observer(observer_name, observer_options)
    suite = Suite(suite_name, suite_instance, suite_options)
    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." % (time.asctime(), ascetime(time.clock() - t0)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            exit(0)
        budget = float(sys.argv[1])
        if observer_options.find('budget') > 0:  # reflect budget in folder name
            idx = observer_options.find('budget')
            observer_options = observer_options[:idx+6] + \
                "%04d" % int(budget + 0.5) + observer_options[idx+10:]
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
    if len(sys.argv) > 4:
        messages = ['Argument "%s" disregarded (only 3 arguments are recognized).' % sys.argv[i]
            for i in range(4, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)
