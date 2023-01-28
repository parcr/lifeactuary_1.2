__author__ = "PedroCR"

import numpy as np
import pandas as pd
from lifeActuary.mortality_table import MortalityTable


class CommutationFunctions(MortalityTable):
    """
    Instantiates, for a specific mortality table and interest rate, all the usual commutation functions:
    Dx, Nx, Sx, Cx, Mx, Rx.

    :param i: interest rate, in percentage. Use 5 for 5% :param g: rate of growing (in percentage), for capitals
    evolving geometrically :param data_type: Use 'l' for lx, 'p' for px and 'q' for qx.
    param mt: the mortality.
    table, in array format, according to the data_type defined.
    param perc: The percentage of qx to use, e.g., use 50 for 50%.
    param app_cont: Use 'True' for continuous approach (deaths occur, in average, in the middle of
    the year and payments are due in the moment of death) or 'False' for considering that death payments are due in
    the end of the year

    :return: the commutation symbols Dx, Nx, Sx, Cx, Mx, Rx.
    """

    def __init__(self, i=None, g=0, data_type='q', mt=None, perc=100, app_cont=False):
        MortalityTable.__init__(self, data_type, mt, perc)
        if i is None:
            return
        self.__i = i / 100.
        self.__g = g / 100.
        self.__v = 1 / (1 + self.__i)
        self.__d = (1 + self.__g) / (1 + self.__i)
        self.__app_cont = app_cont
        self.__cont = np.sqrt(1 + self.__i)

        # self.__Dx = np.array([self.lx[x] * np.power(self__d, x) for x in range(len(self.lx))])
        self.__Dx = self.lx[:-1] * np.power(self.__d, range(len(self.lx[:-1])))
        self.__Nx = np.array([np.sum(self.__Dx[x:]) for x in range(len(self.lx[:-1]))])
        self.__Sx = np.array([np.sum(self.__Nx[x:]) for x in range(len(self.__Nx))])
        self.__Cx = self.dx * np.power(self.__d, range(1, len(self.dx) + 1))
        self.__Mx = np.array([np.sum(self.__Cx[x:]) for x in range(len(self.__Cx))])
        self.__Rx = np.array([np.sum(self.__Mx[x:]) for x in range(len(self.__Mx))])
        if self.__app_cont:
            self.__Cx = self.__Cx * self.__cont
            self.__Mx = self.__Mx * self.__cont
            self.__Rx = self.__Rx * self.__cont

    def __repr__(self):
        return f"{self.__class__.__name__}{self.i, self.g, self.data_type, self.mt, self.perc, self.app_cont}"

    # getters and setters
    @property
    def i(self):
        return self.__i*100

    @property
    def g(self):
        return self.__g*100

    @property
    def v(self):
        return self.__v

    @property
    def d(self):
        return self.__d

    @property
    def app_cont(self):
        return self.__app_cont

    @property
    def cont(self):
        return self.__cont

    @property
    def Dx(self):
        return self.__Dx

    @property
    def Nx(self):
        return self.__Nx

    @property
    def Sx(self):
        return self.__Sx

    @property
    def Cx(self):
        return self.__Cx

    @property
    def Mx(self):
        return self.__Mx

    @property
    def Rx(self):
        return self.__Rx


    def df_commutation_table(self):
        data = {'Dx': self.__Dx, 'Nx': self.__Nx, 'Sx': self.__Sx, 'Cx': self.__Cx, 'Mx': self.__Mx, 'Rx': self.__Rx}
        df = pd.DataFrame(data)
        data_lf = self.df_life_table()
        df = pd.concat([data_lf, df], axis=1, sort=False)
        return df

    ### Life Annuities

    ## Whole Life Annuities

    def ax(self, x, m=1):
        """
        Returns the actuarial present value of an immediate whole life annuity of 1 per year
        Payments of 1/m are made m times per year at the end of the periods.

        :param x: age at the beginning of the contract
        :param m: number of payments per year

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        if x < 0:
            return np.nan
        if m < 0:
            return np.nan
        if x >= self.w:
            return 0
        aux = self.__Nx[x + 1] / self.__Dx[x] / (1 + self.__g) + (m - 1) / (m * 2)
        self.msn.append(f"ax_{x}={self.__Nx[x + 1]}/{self.__Dx[x]}+({m}-1)/({m}*2)")
        return aux

    def aax(self, x, m=1):
        """
        Returns the actuarial present value of a due whole annuity of 1 per year.
        Payments of 1/m are made m times per year at the beginning of the periods.

        :param x: age at the beginning of the contract
        :param m: number of payments per year (used to quote the interest rate)

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        if x > self.w:
            return 1
        aux = self.__Nx[x] / self.__Dx[x] - (m - 1) / (m * 2)
        self.msn.append(f"aax_{x}={self.__Nx[x]}/{self.__Dx[x]}-({m}-1)/({m}*2)")
        return aux

    # Deferred Whole Life Annuities

    def t_ax(self, x, m=1, defer=0):
        """
        Returns the actuarial present value of an (immediate) annuity of 1 per year
        (whole life annuity-late), deferred t years. Payable 'm' times per year at the end of the period. Payable 'm'
        times per year at the end of the period

        :param x: age at the beginning of the contract
        :param m: number of payments per year used to quote the interest rate
        :param defer: deferment period

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        # note: nEx discounts the growth rate np.power(1 + self.__g, defer + 1)
        aux = self.ax(x + defer, m) * self.nEx(x, defer)
        if aux > 0:
            self.msn.append(f"{defer}_ax_{x}=[{self.__Nx[x + 1 + defer]}/{self.__Dx[x + defer]}+({m} + 1)/({m}*2)]"
                            f"*{self.__Dx[x + defer]}/{self.__Dx[x]}")
        return aux

    def t_aax(self, x, m=1, defer=0):
        """
        Returns the actuarial present value of an (immediate) annuity of 1 per year
        (whole life annuity-anticipatory), deferred t years. Payable 'm' times per year at the beginning of the period

        :param x: age at the beginning of the contract
        :param m: number of payments per period used to quote the interest rate
        :param defer: deferment period

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        aux = self.aax(x + defer, m) * self.nEx(x, defer)
        if x + defer < self.w:
            self.msn.append(f"{defer}_aax_{x}=[{self.__Nx[x + defer]}/{self.__Dx[x + defer]}-({m}-1)/({m}*2)]"
                            f"*{self.__Dx[x + defer]}/{self.__Dx[x]}")
        return aux

    ## Temporary Life Annuities

    def nax(self, x, n, m=1):
        """
        Returns the actuarial present value of an immediate temporary life annuity: n-year temporary
        life annuity-late. Payable 'm' times per year at the ends of the period

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        if x >= self.w:
            return 0
        if x < 0:
            return np.nan
        if m < 0:
            return np.nan
        if n < 0:
            return 0

        if x + 1 + n <= self.w:
            aux = (self.__Nx[x + 1] - self.__Nx[x + 1 + n]) / self.__Dx[x] / (1 + self.__g) + \
                  (m - 1) / (m * 2) * (1 - self.nEx(x, n))
            self.msn.append(f"{n}_ax_{x}={self.__Nx[x + 1] - self.__Nx[x + 1 + n]}/{self.__Dx[x]}+({m}-1)/({m}*2)*"
                            f"(1-{self.__Dx[x + n]}/{self.__Dx[x]})")
        else:
            return self.ax(x=x, m=m)

        return aux

    def naax(self, x, n, m=1):
        """
        Returns the actuarial present value of a due temporary life annuity: n-year temporary
        life annuity-anticipatory. Payable 'm' times per year at the beginning of the period

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        if x >= self.w or n == 1:
            return 1
        if x < 0:
            return np.nan
        if m < 0:
            return np.nan
        if n < 0:
            return 0

        if x + 1 + n <= self.w + 1:
            aux = (self.__Nx[x] - self.__Nx[x + n]) / self.__Dx[x] - (m - 1) / (m * 2) * (1 - self.nEx(x, n))
            if x + 1 + n <= self.w:
                Nx2 = self.__Nx[x + 1 + n]
            else:
                Nx2 = 0
            self.msn.append(
                f"{n}_aax_{x}={self.__Nx[x + 1] - Nx2}/{self.__Dx[x]}*(1+{self.__g}) + ({m}+1)/({m}*2)*"
                f"(1-{self.__Dx[x + n]}/{self.__Dx[x]})")
        else:
            return self.aax(x=x, m=m)
        return aux

    # Deferred Temporary Life Annuities

    def t_nax(self, x, n, m=1, defer=0):
        """
        Returns the actuarial present value of a deferred temporary life annuity: n-year temporary
        life annuity-late, deferred t periods. Payable 'm' per year at the ends of the period

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year
        :param defer: deferment period (number of years)

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        aux = self.nax(x + defer, n, m) * self.nEx(x, defer)
        if x + 1 + n + defer <= self.w:
            self.msn.append(
                f"{defer}|{n}_ax_{x}=[{self.__Nx[x + 1 + defer] - self.__Nx[x + 1 + n + defer]}/{self.__Dx[x + defer]}"
                f"+ ({m}-1)/({m}*2)*(1-{self.__Dx[x + n + defer]}/{self.__Dx[x + defer]})]"
                f"*{self.__Dx[x + defer]}/{self.__Dx[x]}")
        else:
            return self.t_ax(x=x, m=m, defer=defer)
        return aux

    def t_naax(self, x, n, m=1, defer=0):
        """
        Returns the actuarial present value of a due temporary life annuity: n-year temporary
        life annuity-anticipatory, deferred t periods. Payable 'm' times per year at the beginning of the period

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year
        :param defer: deferment period (number of years)

        :return: Expected Present Value (EPV) for payments of 1/m
        """
        aux = self.naax(x + defer, n, m) * self.nEx(x, defer)
        if x + 1 + n + defer <= self.w + 1:
            if x + 1 + n + defer <= self.w:
                Nx2 = self.__Nx[x + 1 + n + defer]
            else:
                Nx2 = 0
            self.msn.append(
                f"{defer}|{n}_aax_{x}=[{self.__Nx[x + 1 + defer] - Nx2}/{self.__Dx[x + defer]}"
                f"+({m}+1)/({m}*2)*(1-{self.__Dx[x + n + defer]}/{self.__Dx[x + defer]})]"
                f"*{self.__Dx[x + defer]}/{self.__Dx[x]}")
        else:
            return self.t_aax(x=x, m=m, defer=defer)
        return aux

    ## Life Annuities with variable terms

    # Annuities Increasing and Decreasing Arithmetically

    def t_nIax(self, x, n, m=1, defer=0, first_amount=1, increase_amount=1):
        """
        Returns the actuarial present value of an immediate n term life annuity, deferred $t$ periods,
        with payments evolving in arithmetic progression. Payments of 1/m are made m times per year at the end of the periods.
        First amount and Increase amount may be different.
        For decreasing life annuities, the Increase Amount should be negative.

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year
        :param defer: umber of deferment years
        :param first_amount: amount of the first payment
        :param increase_amount: amount of the increase amount

        :return: Expected Present Value (EPV) for payments of 1/m
        """

        if first_amount + (n - 1) * increase_amount < 0:
            return np.nan
        if x + n + defer > self.w:
            return .0

        term1 = first_amount * self.t_nax(x=x, n=n, m=m, defer=defer)
        list_increases = [increase_amount * self.t_nax(x=x + defer, n=n - j, m=m, defer=defer + j)
                          for j in range(1, n)]

        return term1 + sum(list_increases)

    def t_nIaax(self, x, n, m=1, defer=0, first_amount=1, increase_amount=1):
        """
        Returns the actuarial present value of an immediate n term life annuity, deferred t periods,
        with payments evolving in arithmetic progression. Payments of 1/m are made m times per year at the beginning
        of the periods.
        First amount and Increase amount may be different.
        For decreasing life annuities, the Increase Amount should be negative.

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param m: number of payments per year
        :param defer: umber of deferment years
        :param first_amount: amount of the first payment
        :param increase_amount: amount of the increase amount

        :return: Expected Present Value (EPV) for payments of 1/m
        """

        if first_amount + (n - 1) * increase_amount < 0:
            return np.nan
        if x + n + defer > self.w:
            return .0

        term1 = first_amount * self.t_naax(x=x, n=n, m=m, defer=defer)
        list_increases = [increase_amount * self.t_nax(x=x + defer, n=n-j, m=m, defer=defer + j - 1)
                          for j in range(1, n)]

        return term1+sum(list_increases)

    # Present Value of a series of cash-flows
    def present_value(self, probs, age, spot_rates, capital):
        """
        Computes the expected present value of a cash-flows, that can be contingent on some probabilities.
        Payments are considered at the end of the period.

        :param probs: vector of probabilities. For using the instantiated actuarial table, introduce probs=None
        :param age: age at the beginning of the contract
        :param spot_rates: vector of interest rates for the considered time periods
        :param capital: vector of cash-flow amounts

        :return: the expected present value of a cash-flow, that can be contingent on some probabilities.
        """
        if len(spot_rates) != len(capital):
            return np.nan
        probs_ = None
        if probs is None:
            if age is None:
                return np.nan
            else:
                probs_ = [self.npx(age, n + 1) for n in range(len(capital))]

        if isinstance(probs, (float, int)):
            probs_ = [probs] * len(capital)
        discount = 1 + np.array(spot_rates) / 100.
        discount = np.cumprod(1 / discount)

        return sum([p * capital[idx_p] * discount[idx_p] for idx_p, p in enumerate(probs_)])


    ### Life Insurances

    ## Pure Endowment / Actuarial Present Value

    def nEx(self, x, n):
        """
        Actuarial Present Value of a Pure endowment or Deferred capital

        :param x: age at the beginning of the contract
        :param n: years until payment, if x is alive

        :return: actuarial present value of a pure endowment of 1 paid at age x+n
        """
        if x < 0:
            return np.nan
        if n <= 0:
            return 1
        if x + n > self.w:
            return 0.
        D_x = self.__Dx[x]
        D_x_n = self.__Dx[x + n]
        self.msn.append(f"{n}_E_{x}={D_x_n} / {D_x}")
        # note: nEx discounts the growth rate np.power(1 + self.__g, defer + 1) so only survival is considered
        return D_x_n / D_x / np.power(1 + self.__g, n)

    ## Whole Life Insurance

    def Ax(self, x):
        """
        Expected Present Value (EPV) of a Whole life insurance that pays 1 at the end of the year of death.

        :param x: age at the beginning of the contract

        :return: net single premium of a whole life insurance, that pays 1, at the
        end of the year of death.
        """
        if x < 0:
            return np.nan
        if x > self.w:
            return self.__v  # it will die before year's end, because already attained age>w
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x = self.__Mx[x] / self.__cont
        else:
            M_x = self.__Mx[x]
        self.msn.append(f"A_{x}={M_x} / {D_x}")
        return M_x / D_x / (1 + self.__g)

    def Ax_(self, x):
        """
        Expected Present Value (EPV) of a Whole life insurance that pays 1 at the moment of death.

        :param x: age at the beginning of the contract

        :return: net single premium of a whole life insurance, that pays 1, at the moment of death.
        """
        if x < 0:
            return np.nan
        if x > self.w:  # it will die before year's end, because already attained age>w
            return self.__v ** .5
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x = self.__Mx[x]
        else:
            M_x = self.__Mx[x] * self.__cont
        self.msn.append(f"A_{x}_={M_x} / {D_x}")
        return M_x / D_x / (1 + self.__g)

    # Deferred Whole Life Insurances
    def t_Ax(self, x, defer=0):
        """
        Expected Present Value (EPV) of a Deferred Whole life insurance, that pays 1 in the end of the year of death

        :param x: age at the beginning of the contract
        :param defer: deferment period
        :return: net single premium of a deferred whole life insurance that pays 1, at the
        end of the year of death.
        """
        aux = self.nEx(x, defer) * self.Ax(x + defer)
        self.msn.append(f"{defer}|_A_{x}={defer}_E_{x}*A_{x + defer}")
        return aux

    def t_Ax_(self, x, defer=0):
        """
        Expected Present Value (EPV) of a Deferred Whole life insurance, that pays 1 in the moment of death

        :param x: age at the beginning of the contract
        :param defer: deferment period
        :return: net single premium of a deferred whole life insurance that pays 1, at the moment of death.
        """
        aux = self.nEx(x, defer) * self.Ax_(x + defer)
        self.msn.append(f"{defer}|_A_{x}_={defer}_E_{x}*A_{x + defer}_")
        return aux

    ## Term Life Insurance

    def nAx(self, x, n):
        """
        Expected Present Value (EPV) of a Term life insurance that pays 1 in the end of the year of death, if x dies before age x+n

        :param x: age at the beginning of the contract
        :param n: period of the contract

        :return: net single premium of a Term Life Insurance that pays 1 in the end of the year of death
        """
        if x < 0:
            return np.nan
        if n < 0:
            return np.nan
        if x + n > self.w:
            return self.Ax(x)
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x = self.__Mx[x] / self.__cont
            M_x_n = self.__Mx[x + n] / self.__cont
        else:
            M_x = self.__Mx[x]
            M_x_n = self.__Mx[x + n]
        self.msn.append(f"{n}_A_{x}=({M_x}-{M_x_n}) / {D_x}")
        return (M_x - M_x_n) / D_x / (1 + self.__g)

    def nAx_(self, x, n):
        """
        Expected Present Value (EPV) of a Term life insurance that pays 1 in the moment of death, if x dies before age x+n

        :param x: age at the beginning of the contract
        :param n: period of the contract

        :return: net single premium of a Term Life Insurance that pays 1 in the moment of death
        """
        if x < 0:
            return np.nan
        if n < 0:
            return np.nan
        if x + n > self.w:
            return self.Ax(x) * self.__cont
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x = self.__Mx[x]
            M_x_n = self.__Mx[x + n]
        else:
            M_x = self.__Mx[x] * self.__cont
            M_x_n = self.__Mx[x + n] * self.__cont
        self.msn.append(f"{n}_A_{x}_=({M_x}-{M_x_n}) / {D_x}")
        return (M_x - M_x_n) / D_x / (1 + self.__g)

    # Deferred Term Life Insurances

    def t_nAx(self, x, n, defer=0):
        """
        Expected Present Value (EPV) of a t-years Deferred Term life insurance that pays 1 in the end of the year of death,
        if x dies between ages x+t and x+t+n

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param defer: deferment period (number of years)

        :return: net single premium of a Deferred Term Life Insurance that pays 1 in the end of the year of death
        """
        aux = self.nEx(x, defer) * self.nAx(x + defer, n)
        self.msn.append(f"{defer}|{n}_A_{x}={defer}_E_{x}*{n}_A_{x + defer}")
        return aux

    def t_nAx_(self, x, n, defer=0):
        """
        Expected Present Value (EPV) of a t-years Deferred Term life insurance that pays 1 in the moment of death,
        if x dies between ages x+t and x+t+n

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param defer: deferment period (number of years)

        :return: net single premium of a Deferred Term Life Insurance that pays 1 in the moment of death
        """
        aux = self.nEx(x, defer) * self.nAx_(x + defer, n)
        self.msn.append(f"{defer}|{n}_A_{x}_={defer}_E_{x}*{n}_A_{x + defer}_")
        return aux

    ## Endowment Insurance

    def nAEx(self, x, n):
        """
        Expected Present Value (EPV) of an Endowment Insurance that pays 1 in the end of the year of death, if x dies between
        x and x+n or pays 1 if x is alive at age x+n

        :param x: age at the beginning of the contract
        :param n: period of the contract

        :return: net single premium of an Endowment Insurance. Death coverage is paid at the end of the year of death
        """
        self.msn.append(f"{n}_AE_{x}={n}_A_{x}+{n}_E_{x}")
        return self.nAx(x, n) + self.nEx(x, n)

    def nAEx_(self, x, n):
        """
        Expected Present Value (EPV) of an Endowment Insurance that pays 1 in the moment of death, if x dies between
        x and x+n or pays 1 if x is alive at age x+n

        :param x: age at the beginning of the contract
        :param n: period of the contract

        :return: net single premium of an Endowment Insurance. Death coverage is paid at the moment of death
        """
        aux = self.nAx_(x, n) + self.nEx(x, n)
        self.msn.append(f"{n}_AE_{x}_={n}_A_{x}_+{n}_E_{x}")
        return aux

    # Deferred Endowment Insurance

    def t_nAEx(self, x, n, defer=0):
        """
        Expected Present Value (EPV) of t-years Deferred Endowment Insurance that pays 1 in the end of the year of death,
        if x dies between ages x+t and x+t+n or pays 1 if x is alive at age x+t+n

        :param x: age at the beginning of the contract
        :param n: period of the contract
        :param defer: deferment period (number of years)

        :return: net single premium of a Deferred Endowment Insurance. Death coverage is paid at the end of the year of death
        """
        aux = self.nEx(x, defer) * self.nAEx(x + defer, n)
        self.msn.append(f"{defer}|{n}_AE_{x}={defer}_E_{x}*{n}_AE_{x + defer}")
        return aux

    def t_nAEx_(self, x, n, defer=0):
        """
        Expected Present Value (EPV) of t-years Deferred Endowment Insurance that pays 1 in the moment of death,
        if x dies between ages x+t and x+t+n or pays 1 if x is alive at age x+t+n

        :param x: age at the beginning of the contract
        :param n: period of the contract
        :param defer: deferment period (number of years)

        :return: net single premium of a Deferred Endowment Insurance. Death coverage is paid at the moment of death
        """
        aux = self.nEx(x, defer) * self.nAEx_(x + defer, n)
        self.msn.append(f"{defer}|{n}_AE_{x}={defer}_E_{x}*{n}_AE_{x + defer}_")
        return aux

    ## Term Life Insurance with Variable Capitals

    def IAx(self, x):
        """
        Expected Present Value (EPV) of Term Life Insurance that pays 1+k, at the end of year of death, if death occurs
        between ages x+k and x+k+1, for k=0,1,...

        :param x: age at the beginning of the contract

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the end of the year of death.
        """
        if x < 0:
            return np.nan
        if x > self.w:
            return self.__v  # it will die before year's end, because already attained age>w
        D_x = self.__Dx[x]
        if self.__app_cont:
            R_x = self.__Rx[x] / self.__cont
        else:
            R_x = self.__Rx[x]
        self.msn.append(f"A_{x}={R_x} / {D_x}")
        return R_x / D_x

    def IAx_(self, x):
        """
        Expected Present Value (EPV) of Term Life Insurance that pays 1+k, at the moment of death, if death occurs
        between ages x+k and x+k+1, for k=0,1,...

        :param x: age at the beginning of the contract

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the moment of death.
        """
        if x < 0:
            return np.nan
        if x > self.w:
            return self.__v ** 0.5 # it will die before year's end, because already attained age>w
        D_x = self.__Dx[x]
        if self.__app_cont:
            R_x = self.__Rx[x]
        else:
            R_x = self.__Rx[x] * self.__cont
        self.msn.append(f"A_{x}={R_x} / {D_x}")
        return R_x / D_x

    def nIAx(self, x, n):
        """
        Expected Present Value (EPV) of Term Life Insurance that pays 1+k, at the end of year of death, if death occurs
        between ages x+k and x+k+1, for k=0,1,...,n-1.

        :param x: age at the beginning of the contract

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the end of the year of death.
        """
        if x < 0:
            return np.nan
        if n < 0:
            return np.nan
        if x > self.w:
            return self.__v  # it will die before year's end, because already attained age>w
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x_n = self.__Mx[x + n] / self.__cont
            R_x = self.__Rx[x] / self.__cont
            R_x_n = self.__Rx[x + n] / self.__cont
        else:
            M_x_n = self.__Mx[x + n]
            R_x = self.__Rx[x]
            R_x_n = self.__Rx[x + n]
        self.msn.append(f"A_{x}=({R_x}-{R_x_n}-{n}x{M_x_n} / {D_x}")
        return (R_x - R_x_n - n * M_x_n) / D_x

    def nIAx_(self, x, n):
        """
        Expected Present Value (EPV) of Term Life Insurance that pays 1+k, at the moment of death, if death occurs
        between ages x+k and x+k+1, for k=0,1,...n-1.

        :param x: age at the beginning of the contract

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the moment of death.
        """
        if x < 0:
            return np.nan
        if n < 0:
            return np.nan
        if x > self.w:
            return self.__v ** 0.5 # it will die before year's end, because already attained age>w
        D_x = self.__Dx[x]
        if self.__app_cont:
            M_x_n = self.__Mx[x + n]
            R_x = self.__Rx[x]
            R_x_n = self.__Rx[x + n]
        else:
            M_x_n = self.__Mx[x + n] * self.__cont
            R_x = self.__Rx[x] * self.__cont
            R_x_n = self.__Rx[x + n] * self.__cont
        self.msn.append(f"A_{x}=({R_x}-{R_x_n}-{n}x{M_x_n} / {D_x}")
        return (R_x - R_x_n - n * M_x_n) / D_x

    ## Variable Capitals increasing/decreasing arithmetically

    def nIArx(self, x, n, defer=0, first_amount=1, increase_amount=1):
        """
        Expected Present Value (EPV) of a Term Life Insurance that pays (first_amount + k*increase_amount), at the end of
        the year if death occurs between ages x+k and x+k+1, for k=0,..., n-1.
        Allows the computation of EPV for decreasing capitals.
        The first amount may differ from the increasing/decreasing amount.

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param first_amount: insured amount in the first year of the contract
        :param increase_amount: rate of increasing (if increase_amount>0) or decreasing (if increase_amount<0)

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the end of the year of death.
        """

        if first_amount + (n - 1) * increase_amount < 0:
            return np.nan

        term1 = first_amount * self.t_nAx(x=x, n=n, defer=defer)
        list_increases = [increase_amount * self.t_nAx(x=x + defer, n=n - j, defer=defer + j)
                          for j in range(1, n)]

        return term1 + sum(list_increases)


    def nIArx_(self, x, n, defer=0, first_amount=1, increase_amount=1):
        """
        Expected Present Value (EPV) of a Term Life Insurance that pays (first_amount + k*increase_amount), at the moment of death,
        if death occurs between ages x+k and x+k+1, for k=0,..., n-1.
        Allows the computation of EPV for decreasing capitals.
        The first amount may differ from the increasing/decreasing amount.

        :param x: age at the beginning of the contract
        :param n: number of years of the contract
        :param first_amount: insured amount in the first year of the contract
        :param increase_amount: rate of increasing (if increase_amount>0) or decreasing (if increase_amount<0)

        :return: net single premium of a Term Life Insurance with capitals increasing arithmetically, with first capital
        equal to the rate of progression (the increase amount). The payment is made at the moment of death.
        """

        if first_amount + (n - 1) * increase_amount < 0:
            return np.nan

        term1 = first_amount * self.t_nAx_(x=x, n=n, defer=defer)
        list_increases = [increase_amount * self.t_nAx_(x=x + defer, n=n - j, defer=defer + j)
                          for j in range(1, n)]

        return term1 + sum(list_increases)







