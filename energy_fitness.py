import math


def e_nernst(params):
    return 1.229 - .85 * .001 * (params['T'] - 298.15) + 4.3085 * 10 ** -5 \
           * params['T'](p_h2(params) * math.sqrt(p_o2(params)))


def p_h2(params):
    return 0.5 * params['RHa'] * p_h20_sat(params) * \
           ((1 / (params['RHa'] * p_h20_sat(params) / params['pa'])
             * math.exp((1.635 * (params['i'] / params['A']))(params['T'] ** 1.334))) - 1)


def p_o2(params):
    return params['RHc'] * p_h20_sat(params) * \
           ((1 / (params['RHc'] * p_h20_sat(params) / params['pc'])
             * math.exp((4.192 * (params['i'] / params['A'])) / (params['T'] ** 1.334))) - 1)


def p_h20_sat(params):
    return 2.95 * 10 ** -2 * (params['T'] - 273.15) - 9.19 \
           * 10 ** -5 * (params['T'] - 273.15) ** 2 + 1.44 \
           * 10 ** -7 * (params['T'] - 273.15) ** 3 - 2.18


def n_act(x1, x2, x3, x4, params):
    return -(x1 + x2 * params['T'] + x3 * params['T'] * math.log(c_o2(params)) + x4 * params['T'] * math.log(
        params['i']))


def c_o2(params):
    return p_o2(params) / (5.08 * 10 ** 6 * math.exp(-498 / params['T']))


def n_conc(B, params):
    return - B * math.log(1 - (params['i_den'] / params['i_limit,den']))


def n_ohm(Rc, params):
    return params['i'] * (params['Rm'] + Rc)


def R_m(params):
    return params['sigma_M'] ** params['l'] / params['A']


def sigma_m(lmda, params):
    return (181.6 * (1 + .03 * (params['i'] / params['A']) + .062 * (params['T'] / 303) * (
                params['i'] / params['A']) ** 2.5)) \
           / ((lmda - .634 - 3 * (params['i'] / params['A'])) * math.exp(4.18 * ((params['T'] - 303) / params['T'])))


def v_cell(x, params):
    return e_nernst(params) - n_act(x, params) - n_ohm(params) - n_conc(params)


def v_stack(x, params):
    return params['Ns'] * v_cell(x, params)


def energy_fitness(x, params):
    return (params['V_stack_actual'] - v_stack(x, params))**2
