import jps
import energy


def run(sigma=1/3, seed=None):

    params = {
        'pool_size': 100,
        'cost_target': 0,
        'max_evals': 100,

        # bounds taken from Table 3 in energy paper (Ohenoja-Leiviska)
        'bounds': [
            (-1.19969, -.8532),             # x1
            (.001, .005),                   # x2
            (3.6*10**-5, 9.8*10**-5),       # x3
            (-2.6*10**-4, -9.54*10**-5),    # x4
            (10, 24),                       # lambda
            (.0001, .0008),                 # Rc (Ohms)
            (.0136, .5)                     # B (V)
        ],
        'alpha': 0.999,
        'x': 0.1, # should be .05 - .2  if not good results, then .2 - .3
        'eval_frac': 0.05
    }

    params = {**params, **energy.get_default_params(sigma=sigma, seed=seed)}

    best, best_cost = jps.jps(energy.fitness, params)
    print('{0} => {1}'.format(best, best_cost))


if __name__ == '__main__':
    run()
