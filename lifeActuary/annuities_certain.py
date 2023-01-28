import numpy as np


class Annuities_Certain:

    def __new__(cls, interest_rate, m):
        if interest_rate < 0 or m < 0 or int(m) != m:
            print(f"We need a rate of interest non negative and a positive integer frequency")
            return None
        return object.__new__(cls)

    def __init__(self, interest_rate, m=1):
        '''
        This class instantiates the methods for the computation of financial annuities, for a given
        interest rate and a chosen frequency of payments m (in each period of the interest rate)

        :param interest_rate: interest rate, in percentage (e.g. use 5 for 5%)
        :param m: frequency of payments, in each period of the interest rate

        :return: several methods and annuities for the interest rate and frequency.
        '''
        self.interest_rate = interest_rate / 100.
        self.frequency = m

        self.v = 1 / (1 + self.interest_rate)
        self.im = self.frequency * (np.power(1 + self.interest_rate, 1 / self.frequency) - 1)
        self.vm = np.power((1 + self.im / self.frequency), -1)
        self.dm = self.im * self.vm

    def check_terms(func):
        def func_wrapper(self, terms, *args, **kwargs):
            if not terms:
                terms=0
            if terms < 0 or int(terms) != terms:
                return np.nan
            res = func(self, terms)
            return res

        return func_wrapper

    def check_grow(func):
        def func_wrapper(self, terms, payment, grow):
            if grow / 100 <= -1 or terms < 0 or int(terms) != terms:
                return np.nan
            res = func(self, terms, payment, grow)
            return res

        return func_wrapper

    # Constant Term Financial Annuities

    @check_terms
    def an(self, terms):
        '''
        Returns the present value of an immediate n-term financial annuity with payments equal to 1.
        Payments are made in the end of the periods. In fractional annuities, payments of 1/m are made
        m times per year at the end of the periods.

        :param terms: number of years

        :return: Expected Present Value (EPV) of an immediate n-term financial annuity
        '''
        if not terms:
            return 1 / self.im
        return (1 - np.power(self.vm, terms * self.frequency)) / self.im

    @check_terms
    def aan(self, terms):
        '''
        Returns the present value of a due n-term financial annuity with payments equal to 1.
        Payments are made in the end of the periods. In fractional annuities, payments of 1/m are made
        m times per year at the end of the periods.

        :param terms: number of years

        :return: Expected Present Value (EPV) of a due n-term financial annuity
        '''
        if not terms:
            return 1 / self.dm
        return (1 - np.power(self.vm, terms * self.frequency)) / self.dm

    # Variable Terms Financial Annuities
    @check_terms
    def Ian(self, terms, payment=1, increase=1):
        '''
        Returns the present value of an immediate $n$ term financial annuity with payments
        increasing/decreasing arithmetically. Payments are made in the end of the periods.
        First payment and increase amount may differ.
        In fractional annuities, payments level within each interest period and increase/decrease from one
        interest period to the next.

        :param terms: number of years
        :param payment: amount of the first payment
        :param increase: increase amount of payments (positive for increasing annuities and negative for decreasing annuities)

        :return: Expected Present Value (EPV) of an immediate arithmetically increasing/decreasing financial annuity. Payments level within each interest period and increase/decrease from one interest period to the next.
        '''
        if payment + increase * terms < 0:
            return np.nan
        # (payment - increase) * self.an(terms) + increase * (self.aan(terms) - terms * self.v ** terms) / self.im
        return payment * self.an(terms) + increase / self.im * (
                    (1 - self.v ** terms) / self.interest_rate - terms * self.v ** terms)

    @check_terms
    def Iaan(self, terms, payment=1, increase=1):
        '''
        Returns the present value of a due $n$ term financial annuity with payments increasing/decreasing arithmetically. Payments are made in the beginning of the periods.
        First payment and increase amount may differ.
        In fractional annuities, payments level within each interest period and increase/decrease from one
        interest period to the next.

        :param terms: number of years
        :param payment: amount of the first payment
        :param increase: increase amount of payments (positive for increasing annuities and negative for decreasing annuities)

        :return: Expected Present Value (EPV) of a due arithmetically increasing/decreasing financial annuity. Payments level within each interest period and increase/decrease from one interest period to the next
        '''
        return self.Ian(terms, payment, increase) / self.vm

    @check_terms
    def Iman(self, terms, payment=1, increase=1):
        '''
        Returns the present value of an immediate $n$-term financial annuity with payments
        increasing/decreasing arithmetically.
        Payments are made in the end of the periods.
        First payment and increase amount may differ.
        In fractional annuities, payments increase in each payment period.

        :param terms: number of years
        :param payment: amount of the first payment
        :param increase: increase amount of payments (positive for increasing annuities and negative for decreasing annuities)

        :return: Expected Present Value (EPV) of an arithmetically increasing/decreasing financial annuity.
        Payments increase in each payment period and are paid in the end of periods.
        '''
        if payment + increase * terms < 0:
            return np.nan

        return (payment - increase) * self.an(terms) \
               + increase * self.v \
               * (self.v ** terms * ((terms * self.frequency) * (self.vm - 1) - 1) + 1) \
               / (self.frequency * self.v ** ((self.frequency - 1) / self.frequency) * (self.vm - 1) ** 2)

    @check_terms
    def Imaan(self, terms, payment=1, increase=1):
        '''
        Returns the present value of an immediate $n$-term financial annuity with payments
        increasing/decreasing arithmetically.
        Payments are made in the beginning of the periods.
        First payment and increase amount may differ.
        In fractional annuities, payments increase in each payment period.

        :param terms: number of years
        :param payment: amount of the first payment
        :param increase: increase amount of payments (positive for increasing annuities and negative for decreasing annuities)

        :return: Expected Present Value (EPV) of an arithmetically increasing/decreasing financial annuity.
        Payments increase in each payment period and are paid in the beginning of periods.
        '''

        return self.Iman(terms, payment, increase) / self.vm

    @check_grow
    def Gan(self, terms, payment=1, grow=0):
        '''
        Returns the present value of an immediate $n$ term financial annuity with payments increasing/decreasing geometrically. Payments are made in the end of the periods. In fractional annuities, payments level within each interest period and increase/decrease from one interest period to the next.

        :param terms: number of years
        :param payment: amount of the first payment
        :param grow: growth rate from one year to the other

        :return: Expected Present Value (EPV) of an arithmetically increasing/decreasing financial annuity. Payments increase in each year.
        '''

        v = (1 + grow / 100) * self.v
        if self.interest_rate == grow / 100:
            return payment * terms * self.frequency * self.vm / self.frequency
        return payment / (1 + grow / 100) ** (1 / self.frequency) * (1 - v ** terms) / \
               (1 - v ** (1 / self.frequency)) * v ** (1 / self.frequency) / self.frequency

    @check_grow
    def Gaan(self, terms, payment=1, grow=0):
        '''
        Returns the present value of a due $n$ term financial annuity with payments increasing/decreasing geometrically. Payments are made in the beginning of the periods. In fractional annuities, payments level within each interest period and increase/decrease from one interest period to the next.

        :param terms: number of years
        :param payment: amount of the first payment
        :param grow: growth rate from one year to the other

        :return: Present Value of an arithmetically increasing/decreasing financial annuity. Payments increase in each year.
        '''
        v = (1 + grow / 100) * self.v
        return self.Gan(terms, payment, grow) / v ** (1 / self.frequency)

    @check_grow
    def Gman(self, terms, payment=1, grow=0):
        '''
        Returns the present value of an immediate $n$ term financial annuity with payments increasing/decreasing geometrically. Payments are made in the end of the periods. In fractional annuities, payments increase in each payment period.

        :param terms: number of years
        :param payment: amount of the first payment
        :param grow: rate of growing of payments, in percentage

        :return: Present Value of an arithmetically increasing/decreasing financial annuity with terms paid in the end of periods.
        Payments increase in each payment period.
        '''
        a1 = (1 - self.v) / self.im
        if self.interest_rate == grow / 100:
            return a1 * terms
        ig = (self.interest_rate - grow / 100) / (1 + grow / 100)
        vg = 1 / (1 + ig)
        a2 = (1 - vg ** terms) / (1 - vg)
        return payment * a1 * a2

    @check_grow
    def Gmaan(self, terms, payment=1, grow=0):
        '''
        Returns the present value of an immediate $n$ term financial annuity with payments increasing/decreasing geometrically. Payments are made in the beginning of the periods. In fractional annuities, payments increase in each payment period.

        :param terms: number of years
        :param payment: amount of the first payment
        :param grow: rate of growing of payments, in percentage

        :return: Present Value of an arithmetically increasing/decreasing financial annuity with terms paid in the beginning of periods.
        Payments increase in each payment period.
        '''
        v = (1 + grow / 100) * self.v
        return self.Gman(terms, payment, grow) / v ** (1 / self.frequency)
