import random
import numpy as np


def mutate_dim(dim, hi, lo, params, u=random.uniform, p=2):
    """
    Mutate a dimension. If the mutation falls outside of the allowed bounds for this dimension, clip the value to its
    upper or lower bound which ever is closest to the mutate value.

    :param dim: (float) the dimension to mutate
    :param hi: (float) the upper bound of this dimension
    :param lo: (float) the lower bound of this dimension
    :param params: (dict) the user provided parameters for this run (see comments above for more details
    :param u: (function) the function to generate a uniform random number between 0 and 1.
    :param p: (int) precision for rounding the mutated dimension's value
    :return: (float) the mutated value for this dimension
    """
    # mutate this dimension with 50/50 chance of velocity being positive
    v = params['x'] * (hi - lo) * u(0, 1) if random.choice([0, 1]) is 0 else -(params['x'] * (hi - lo) * u(0, 1))
    dim_mutated = dim + v

    # clip mutation if value is out of bounds
    if dim_mutated < lo:
        dim_mutated = lo if dim is lo else lo + u(0, 1) * (dim - lo)
    elif dim_mutated > hi:
        dim_mutated = hi if dim is hi else hi - u(0, 1) * (hi - dim)

    return round(dim_mutated, p)


def mutate(member, params, u=random.uniform, p=2):
    """
    Mutate a pool member (attribute vector). If the mutated value of any given dimension falls outside of the its
    allowed bounds, clip said dimension to its upper or lower bounds, which ever is closest to the mutated value.

    :param member: (tuple) the pool member to mutate. Each attribute is the tuple is assumed to be a continuous value
        within the upper and lower bounds for that particular attribute.
    :param params: (dict) the user provided parameters for this run (see comments above for more details).
    :param u: (function) the function to generate a uniform random number between 0 and 1.
    :param p: (int) precision for rounding each attribute for the mutated member
    :return: (tuple) the mutated member (attribute vector)
    """
    return tuple(mutate_dim(member[i], params['bounds'][i][0], params['bounds'][i][1], params, u, p)
                 for i in range(len(member)))


def gen_pool(cost, params, u=random.uniform, p=2):
    """
    Generate an initial population (member pool) (comprised of half randomly generated members, and the other half from
    mutations of the the orginal randomly generated members), and determine the average (positive) cost difference
    between each randomly generated member and its corresponding mutation.

    :param cost: (function) the objective function
    :param params: (dict) the user provided parameters for this run (see comments above for more details).
    :param u: (function) the function to generate a uniform random number between 0 and 1.
    :param p: (int) precision for rounding the each attribute for each randomly generated and mutated member.
    :return: (tuple) 2-dimensional tuple where the first element (list) is the initial population (member pool) and the
        second element (float) is the average (positive) cost difference between each randomly generated member and its
        corresponding mutation.
    """
    def gen_member():
        return tuple(round(u(lo, hi), p) for lo, hi in params['bounds'])

    # generate half of the pool randomly and mutate each member for the remaining half of the pool
    rand_pool_half = [gen_member() for _ in range(int(params['pool_size']/2))]
    mutant_pool_half = [mutate(t, params, u, p) for t in rand_pool_half]

    # determine the difference average of the pool determining the (positive) cost difference for each trail solution
    # and its corresponding mutant in the pool
    davg = np.mean([cost(m) - cost(t) for t in rand_pool_half for m in mutant_pool_half])

    return rand_pool_half + mutant_pool_half, davg


def prob(d, davg, evals, params, p=2):
    """
    Determines the probability to accept a "worse" solution vs greedly taking the best solution. This helps with
    performing simulated annealing in the DE algorithm.

    :param d: (float) the (positive) cost difference between a member vector and its mutant vector
    :param davg: (float) the average (positive) cost difference between each randomly generated member and its
        corresponding mutated member for each member that was generated in the initial pool (population)
    :param evals: (int) the number of evaluations we have performed thus far
    :param params: (dict) the user provided parameters for this run (see comments above for more details).
    :param p: (int) precision for rounding the probability
    :return: (float) the probability we should use in determining if we choose the worst over the best solution
    """
    return round(params['alpha']**evals, p) if evals <= (params['eval_frac'] * params['max_evals']) \
        else round(params['alpha']**evals / (1 + (d/davg)**2), p)


def de(cost, params, s=None, p=2):
    """
    Execute a Differential Evolution run

    :param cost: (function) the objective function
    :param params: (dict) the user provided parameters for this run (see comments above for more details).
    :param s: (int) seed used for generating pseudorandom number
    :param p: (int) precision used for rounding all attributes for all members throughout the entire run
    :return: (tuple) 2-dimensional tuple
    """
    random.seed(s)
    np.random.seed(s)

    evals = 0
    pool, davg = gen_pool(cost, params, u, p)
    # determine the cost of each member in the pool and remember the best member and its corresponding cost
    best = sorted([(m, cost(m)) for m in pool], key=lambda tup: tup[1])[0][0]
    for i, current in enumerate(pool):
        if cost(best) < params['cost_target'] or evals > params['max_evals']:
            break
        next = mutate(current, params, u)
        if cost(next) < cost(best):
            best = next

        if cost(next) < cost(current):
            pool[i] = next
        else:
            d = cost(next) - cost(current)
            p = prob(d, davg, evals, params)
            if np.random.choice([1, 0], p=[p, 1-p]) is 1:
                pool[i] = next

        evals += 1

    return best, cost(best)


if __name__ == '__main__':
    params = {
        'pool_size': 1000,
        'cost_target': 0,
        'max_evals': 100000,
        'bounds': [(-10, 10), (-10, 10)],
        'alpha': 0.999,
        'x': 0.2,
        'eval_frac': 0.05
    }

    def cost(m):
        return m[0]**2 + m[1]**2

    print(de(cost, params))

