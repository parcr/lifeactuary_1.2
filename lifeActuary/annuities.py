__author__ = "PedroCR"

import numpy as np


# life generic annuity 1 head
def annuity_x(mt, x, x_first, x_last, i=None, g=.0, m=1, method='udd'):
    '''
    Computes the present value of an annuity that starts paying 1 at age x, increasing by (1+g/100) and stops
    at age x_w, paying (1+g/100)^{x_w-x}
    :param mt: table for life x
    :param x: age x
    :param x_first: age of first payment
    :param x_last: age of final payment
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x_first < x: return np.nan
    if x_last < x_first == x: return np.nan
    if int(m) != m: return np.nan
    if x == x_first == x_last: return 1
    i = i / 100
    g = g / 100
    d = float((1 + g) / (1 + i))
    number_of_payments = int((x_last - x_first) * m + 1)
    payments_instants = np.linspace(x_first - x, x_last - x, number_of_payments)
    instalments = [mt.npx(x, n=t, method=method) *
                   np.power(d, t) for t in payments_instants]
    instalments = np.array(instalments) / np.power(1 + g, x_first - x) / m
    return np.sum(instalments)


# life annuities_1 1 head
# immediate
def ax(mt, x, i=None, g=0, m=1, method='udd'):
    '''
    Returns a whole life annuity immediate
    :param mt: table for life x
    :param x: age x
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + 1 / m > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + 1 / m, x_last=mt.w, i=i, g=g, m=m, method=method)


def t_ax(mt, x, i=None, g=0, m=1, defer=0, method='udd'):
    '''
    Returns a whole life annuity immediate, deferred
    :param mt: table for life x
    :param x: age x
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + 1 / m + defer > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + 1 / m + defer, x_last=mt.w, i=i, g=g, m=m, method=method)


def nax(mt, x, n, i=None, g=0, m=1, method='udd'):
    '''
    Return the actuarial present value of a (immediate) temporal (term certain) annuity: n-year temporary
    life annuity-late. Payable 'm' per year at the ends of the period
    :param mt: table for life x
    :param x: age x
    :param n: total amount payed
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + 1 / m > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + 1 / m, x_last=x + n, i=i, g=g, m=m, method=method)


def t_nax(mt, x, n, i=None, g=0, m=1, defer=0, method='udd'):
    '''
    Return the actuarial present value of a (immediate) temporal (term certain) annuity: n-year temporary
    life annuity-late. Payable 'm' per year at the ends of the period, deferred
    :param mt: table for life x
    :param x: age x
    :param n: total amount payed
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + 1 / m + defer > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + 1 / m + defer, x_last=x + n + defer, i=i, g=g, m=m, method=method)


# due
def aax(mt, x, i=None, g=0, m=1, method='udd'):
    '''
    Returns a whole life annuity due
    :param mt: table for life x
    :param x: age x
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x > mt.w: return 1

    return annuity_x(mt=mt, x=x, x_first=x, x_last=mt.w, i=i, g=g, m=m, method=method)


def t_aax(mt, x, i=None, g=0, m=1, defer=0, method='udd'):
    '''
    Returns a whole life annuity due, deferred
    :param mt: table for life x
    :param x: age x
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + defer > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + defer, x_last=mt.w, i=i, g=g, m=m, method=method)


def naax(mt, x, n, i=None, g=0, m=1, method='udd'):
    '''
    Return the actuarial present value of a (due) temporal (term certain) annuity: n-year temporary
    life annuity-due. Payable 'm' per year at the ends of the period
    :param mt: table for life x
    :param x: age x
    :param n: total amount payed
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x > mt.w: return 1

    return annuity_x(mt=mt, x=x, x_first=x, x_last=x + n - 1 / m, i=i, g=g, m=m, method=method)


def t_naax(mt, x, n, i=None, g=0, m=1, defer=0, method='udd'):
    '''
    Return the actuarial present value of a (due) temporal (term certain) annuity: n-year temporary
    life annuity-due. Payable 'm' per year at the ends of the period, deferred
    :param mt: table for life x
    :param x: age x
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param m: frequency of payments per unit of interest rate quoted
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the actuarial present value
    '''
    if x + defer > mt.w: return 0

    return annuity_x(mt=mt, x=x, x_first=x + defer, x_last=x + n + defer - 1 / m, i=i, g=g, m=m, method=method)


def nEx(mt, x, i=None, g=0, defer=0, method='udd'):
    """
    Pure endowment or Deferred capital
    :param x: age at the beginning of the contract
    :param i: technical interest rate (flat rate) in percentage, e.g., 2 for 2%
    :param g: growth rate (flat rate) in percentage, e.g., 2 for 2%
    :param defer: deferment period
    :param method: the method to approximate the fractional periods
    :return: the present value of a pure endowment of 1 at age x+n
    """

    return t_naax(mt=mt, x=x, n=1, i=i, g=g, m=1, defer=defer, method=method)
