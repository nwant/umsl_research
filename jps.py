import random
import numpy as np


def mutate_dim(dim, hi, lo, u=random.uniform):
    # mutate this dimension with 50/50 chance of velocity being positive
    v = params['x'] * (hi - lo) * u(0, 1) if random.choice([0, 1]) is 0 else -(params['x'] * (hi - lo) * u(0, 1))
    dim_mutated = dim + v

    # clip mutation if value is out of bounds
    if dim_mutated < lo:
        dim_mutated = lo if dim is lo else lo + u(0, 1) * (dim - lo)
    elif dim_mutated > hi:
        dim_mutated = hi if dim is hi else hi - u(0, 1) * (hi - dim)

    return dim_mutated


def mutate(member, params, u=random.uniform):
    return tuple(mutate_dim(member[i], hi, lo, u) for i in range(len(member)) for lo, hi in params['bounds'][i])


def gen_pool(cost, params, u=random.uniform, p=2):
    def gen_member():
        return tuple(round(u(lo, hi), p) for lo, hi in params['bounds'])

    # generate half of the pool randomly and mutate each member for the remaining half of the pool
    rand_pool_half = [gen_member() for _ in range(int(params['pool_size']/2))]
    mutant_pool_half = [mutate(t, params, u) for t in rand_pool_half]

    # determine the difference average of the pool determining the (positive) cost difference for each trail solution
    # and its corresponding mutant in the pool
    davg = np.mean([cost(m) - cost(t) for t in rand_pool_half for m in mutant_pool_half])

    return rand_pool_half + mutant_pool_half, davg


def prob(d, darg, evals, params, p=2):
    return round(params['alpha']**evals, p) if params['evals'] <= params['eval_frac'] * params['max_evals'] \
        else round(params['alpha']**evals / (1 + (d/darg)**2), p)


def min_cost(cost, params, u=random.uniform, s=None, p=2):
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
            if np.random.choices([1, 0], p=[p, 1-p]) is 1:
                pool[i] = next

        evals += 1

    return best


if __name__ == '__main__':
    params = {
        'pool_size': 10,
        'cost_target': 0,
        'max_evals': 100,
        'bounds': [(-10, 10), (-10, 10)],
        'alpha': 0.999,
        'x': 0.2,
        'eval_frac': 0.05
    }

    def cost(m):
        return m[0]**2 + m[1]**2

    print(min_cost(cost, params, s=2))

