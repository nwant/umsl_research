# Nathaniel Want
# May 25, 2018
import math


def e_nernst(i, params):
    """
    Determine the thermodynamic potential (aka emf, Nernst voltage, or reversible voltage) of a PEM fuel cell

    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the thermodynamic potential of the PEM fuel cell, in volts
    """
    return 1.229 - .85 * .001 * (params['T'] - 298.15) + 4.3085 * 10 ** -5 * params['T'](p_h2(i, params) *
                                                                                         math.sqrt(p_o2(i, params)))


def p_h2(i, params):
    """
    Determine the partial pressure change of hydrogen relative to the current of a PEM cell

    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the partial pressure change of hydrogen, in atm
    """
    return 0.5 * params['RHa'] * p_h20_sat(params) * \
           ((1 / (params['RHa'] * p_h20_sat(params) / params['p_a']) *
             math.exp((1.635 * (i / params['A']))(params['T'] ** 1.334))) - 1)


def p_o2(i, params):
    """
    Determine the partial pressure change of oxygen relative to the current of a PEM cell

    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the partial pressure change of oxygen, in atm
    """
    return params['RHc'] * p_h20_sat(params) * \
           ((1 / (params['RHc'] * p_h20_sat(params) / params['p_c'])
             * math.exp((4.192 * (i / params['A'])) / (params['T'] ** 1.334))) - 1)


def p_h20_sat(params):
    """
    Determine the saturation pressure of water

    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the saturation pressure of water, in atm
    """
    return 10 ** (2.95 * 10 ** -2 * (params['T'] - 273.15) - 9.19
                  * 10 ** -5 * (params['T'] - 273.15) ** 2 + 1.44
                  * 10 ** -7 * (params['T'] - 273.15) ** 3 - 2.18)


def n_act(x1, x2, x3, x4, i, params):
    """
    Determine the activation loss caused by the slowness of the electrochemical reactions taking place on the surface
    of the electrode.

    :param x1: (float) parametric coefficient for the PEM fuel cell
    :param x2: (float) parametric coefficient for the PEM fuel cell
    :param x3: (float) parametric coefficient for the PEM fuel cell
    :param x4: (float) parametric coefficient for the PEM fuel cell
    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the activation loss
    """
    def c_o2(i, params):
        return p_o2(i, params) / (5.08 * 10 ** 6 * math.exp(-498 / params['T']))

    return -(x1 + x2 * params['T'] + x3 * params['T'] * math.log(c_o2(i, params)) + x4 * params['T'] * math.log(i))


def n_conc(B, i, params):
    """
    Determine the concentration loss (aka mass transport loss) due to the change in concentration of the reactants at
    the surface of the electrodes as the fuel is used.

    :param B: (float) a parametric coefficient for the PEM fuel cell, in volts
    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the concentration (or mass transport) loss
    """
    return - B * math.log(1 - ((i / params['A']) / params['i_limit,den']))


def n_ohm(lmda, Rc, i, params):
    """
    Determine the Ohmic loss due to electrical resistance of the electrodes, the polymer membrane, and the conducting
    resistance between the membrane and the electrodes.

    :param lmda: (float) a parametric coefficient for the PEM fuel cell
    :param Rc: (float) the resistance (assumed constant) to the transfer of protons through the membrane
    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the Ohmic loss
    """
    def R_m(lmda, i, params):
        return (sigma_m(lmda, i, params) ** params['l']) / params['A']

    def sigma_m(lmda, i, params):
        return (181.6 * (1 + .03 * (i / params['A']) + .062 * (params['T'] / 303) * (i / params['A']) ** 2.5)) \
           / ((lmda - .634 - 3 * (i / params['A'])) * math.exp(4.18 * ((params['T'] - 303) / params['T'])))

    return i * (R_m(lmda, i, params) + Rc)


def v_cell(x1, x2, x3, x4, lmda, Rc, B, i, params):
    """
    Determine the terminal voltage of a single PEM cell, not considering loss due to fuel crossover and internal
    currents.

    :param x1: (float) parametric coefficient for the PEM fuel cell
    :param x2: (float) parametric coefficient for the PEM fuel cell
    :param x3: (float) parametric coefficient for the PEM fuel cell
    :param x4: (float) parametric coefficient for the PEM fuel cell
    :param lmda: (float) a parametric coefficient for the PEM fuel cell
    :param Rc: (float) the resistance (assumed constant) to the transfer of protons through the membrane
    :param B: (float) a parametric coefficient for the PEM fuel cell, in volts
    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the terminal voltage of the cell
    """
    return e_nernst(params, i) - n_act(x1, x2, x3, x4, i, params) - n_ohm(lmda, Rc, i, params) - n_conc(B, i, params)


def v_stack(x, i, params):
    """
    Determine the terminal voltage of a stack

    :param x: (tuple) the attribute vector for the PEMFC model.
        Should contain 7 attributes in the following order: (x1, x2, x3, x4, lambda, Rc, B)
    :param i: (float) the current (Amperes) of the PEM cell
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the terminal voltage of a stack
    """
    return params['Ns'] * v_cell(x[0], x[1], x[2], x[3], x[4], x[5], x[6], i, params)


def get_default_params():
    """
    Get the stack parameters as defined in the "PEM fuel cell modeling using differential evolution" paper

    Contains the following key-values:
    'Ns': (int) the number of series cells in a stack
    'A': (float) the cell area (effective electrode area) (cm^2)
    'l': (float) the membrane thickness (nm)
    'i_limit,den': (float) the limiting current density (A/cm^2)
    'RHa': (float) the relative humidity of vapor in anode
    'RHc': (float) the relative humidity of vapor in cathode
    'T': (float) the stack temperature (K)
    'p_a': (float) anode inlet pressure (atm)
    'p_c': (float) cathode inlet pressure (atm)

    :return: (dict) the default stack parameters
    """
    return {
        'Ns': 24,
        'A': 27 * 10 ** -2,
        'l': 127 * 10 ** -9,
        'i_limit,den': 860,
        'RHa': 1,
        'RHc': 1,
        'T': 353.15,
        'p_a': 3,
        'p_c': 5
    }


def energy_fitness(x, j, params=get_default_params()):
    """
    Determine the fitness for a given attribute vector
    :param x: (tuple) the attribute vector for the PEMFC model to determine the fitness for.
        Should contain 7 attributes in the following order: (x1, x2, x3, x4, lambda, Rc, B)
    :param j: (list) a list of 2 dimensional tuples, that provide the current-voltage pairs (current being the first
        element in the tuple, and voltage being the second).
    :param params: (dict) various stack parameters and operating conditions (see section on params)
    :return: (float) the fitness of the attribute vector (x)
    """
    # j = [1.1, 2.5, 3.9, 5.3, 6.7, 8.1, 9.5, 10.9, 12.3, 13.7, 15.1, 16.5, 17.9, 19.3, 20.7, 22.1]
    return sum([(v - v_stack(x, i, params)) ** 2 for i, v in j])


