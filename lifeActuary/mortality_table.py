__author__ = "PedroCR"

import numpy as np
import pandas as pd


class MortalityTable:
    """
    Instantiates a life table, where you can pass it in the form of the lx, qx or px. Note that the first value is
    the first age considered in the table.
    The life table will be complete, that is, from age 0 to age w, that is, the last age where lx>0.
    """

    def __init__(self, data_type='q', mt=None, perc=100, last_q=1):
        """
        Initializes the MortalityTable class, so we can construct a mortality table with the usual fields.
        param data_type: Should be "l" for lx, "p" for px and "q" for qx.
        :param mt: Should be "l" for lx, "p" for px and "q" for qx.
        :param perc: The percentage of qx to use, e.g., you should use 50 for 50%.
        :param last_q: The value for qw.
        """
        if data_type not in ('l', 'q', 'p'):
            return
        if not mt:
            return
        self.__data_type = data_type
        self.__methods = ('udd', 'cfm', 'bal')
        self.__mt = mt
        self.__x0 = np.int(mt[0])
        self.__last_q = last_q
        self.__w = 0
        self.__lx = []
        self.__px = []
        self.__qx = []
        self.__dx = []
        self.__ex = []
        self.__perc = perc
        self.msn = []

        radical = 100000.
        pperc = perc / 100.
        mt = np.array(mt[1:])
        if data_type == 'l':
            if mt[-1] > 0:
                mt = np.append(mt, 0)
            self.__qx = (mt[:-1] - mt[1:]) / mt[:-1] * pperc
            # self.__qx = np.append(np.zeros(self.x0), self.__qx)
        if data_type == 'q':
            self.__qx = mt * pperc
        if data_type == 'p':
            self.__qx = (1 - mt) * pperc

        if self.__last_q == 1 and self.__qx[-1] < 1 - .1e-10:
            self.__qx = np.append(self.__qx, 1)
        if self.__last_q == 0 and self.__qx[-1] > .1e-10:
            self.__qx = np.append(self.__qx, 0)
        self.__qx = np.append(np.zeros(self.x0), self.__qx)

        self.__px = 1 - self.__qx
        self.__lx = np.array([radical] * (len(self.__qx) + 1))
        self.__dx = np.array([-1] * len(self.__qx))
        self.__ex = np.array([-1] * len(self.__qx))
        for idx_p, p in enumerate(self.__px):
            self.__lx[idx_p + 1] = self.__lx[idx_p] * p
        self.__dx = self.__lx[:-1] * self.__qx
        sum_lx = np.array([sum(self.__lx[l:]) for l in range(len(self.__qx))])
        self.__ex = sum_lx[1:] / self.__lx[:-2]
        self.__ex = np.append(self.__ex, 0) + .5
        self.__w = len(self.__lx) - 2

    def __repr__(self):
        return f"{self.__class__.__name__}{self.data_type, self.mt, self.__perc, self.__last_q}"

    # getters and setters
    @property
    def data_type(self):
        return self.__data_type

    @property
    def w(self):
        return self.__w

    @property
    def mt(self):
        return self.__mt

    @property
    def methods(self):
        return self.__methods

    @property
    def x0(self):
        return self.__x0

    @property
    def lx(self):
        return self.__lx

    @property
    def px(self):
        return self.__px

    @property
    def qx(self):
        return self.__qx

    @property
    def dx(self):
        return self.__dx

    @property
    def ex(self):
        return self.__ex

    @property
    def perc(self):
        return self.__perc

    def df_life_table(self):
        data = {'x': np.arange(self.w + 1), 'lx': self.__lx[:-1], 'dx': self.__dx,
                'qx': self.__qx, 'px': self.__px, 'exo': self.__ex}
        df = pd.DataFrame(data)
        df = df.astype({'x': 'int16'})
        return df

    def lx_udd(self, t):
        if t > self.w+1:
            return 0.
        if t < 0:
            return np.nan
        int_t = np.int(t)
        frac_t = t - int_t
        if frac_t == 0:
            return self.__lx[int_t]
        else:
            return self.__lx[int_t] * (1 - frac_t) + self.__lx[int_t + 1] * frac_t

    def lx_cfm(self, t):
        if t > self.w+1:
            return 0.
        if t < 0:
            return np.nan
        int_t = np.int(t)
        frac_t = t - int_t
        if frac_t == 0:
            return self.__lx[int_t]
        else:
            return self.__lx[int_t] * np.power(self.__lx[int_t + 1] / self.__lx[int_t], frac_t)

    def lx_bal(self, t):
        if t > self.w+1:
            return 0.
        if t < 0:
            return np.nan
        int_t = np.int(t)
        frac_t = t - int_t
        if frac_t == 0:
            return self.__lx[int_t]
        else:
            inv_lx = 1 / self.__lx[int_t] - frac_t * (1 / self.__lx[int_t] - 1 / self.__lx[int_t + 1])
            return 1 / inv_lx

    def get_lx_method(self, x, method='udd'):
        if method not in self.__methods:
            return np.nan
        if x < 0:
            return np.nan
        if x > self.w+1:
            return 0
        if method == 'udd':
            return self.lx_udd(x)
        elif method == 'cfm':
            return self.lx_cfm(x)
        elif method == 'bal':
            return self.lx_bal(x)
        else:
            return np.nan

    def get_integral_px_method(self, x, method='udd'):
        '''
        This function can be used to approximate the life expectancy between to ages, using the interpolation rules implemented.
        :param x: the value of x, that should be integer
        :param method: the chosen method to approximate px
        :return: the integral of tpx in the interval [0,1] that can be used to approximate life expectancy between ages.
        '''
        if method not in self.__methods:
            return np.nan
        if x < 0:
            return np.nan
        if x > self.w+1:
            return 0
        if int(x) != x:
            return np.nan
        if method == 'udd':
            return 1 - .5 * self.qx[x]
        elif method == 'cfm':
            if self.px[x] == 0:
                return .0
            return -self.qx[x] / np.log(self.px[x])
        elif method == 'bal':
            if self.px[x] == 0:
                return .0
            return -self.px[x] / self.qx[x] * np.log(self.px[x])
        else:
            return np.nan

    def nqx(self, x, n=1, method='udd'):
        '''
        Obtains the probability that a life x dies before x+t
        :param method: the method used to approximate lx for non-integer x's
        :param x: age at beginning
        :param n: period
        :return: probability of x dying before x+t
        '''
        if method not in self.__methods:
            return np.nan
        if x < 0:
            return np.nan
        if n <= 0:
            return .0
        if x + n > self.w+1:
            return self.__qx[-1]
        l_x = self.get_lx_method(x, method)
        l_x_t = self.get_lx_method(x + n, method)
        self.msn.append(f"{n}_q_{x}=1-({l_x_t} / {l_x})")
        return 1 - l_x_t / l_x

    def npx(self, x, n=1, method='udd'):
        '''
        Obtains the probability that a life x dies before x+t
        :param method: the method used to approximate lx for non-integer x's
        :param x: age at beginning
        :param n: period
        :return: probability of x dying before x+t
        '''
        if method not in self.__methods:
            return np.nan
        if x < 0:
            return np.nan
        if n <= 0:
            return 1.
        if x + n > self.w+1:
            return self.__px[-1]
        l_x = self.get_lx_method(x, method)
        l_x_t = self.get_lx_method(x + n, method)
        self.msn.append(f"{n}_p_{x}={l_x_t} / {l_x}")
        return l_x_t / l_x

    def t_nqx(self, x, t=1, n=1, method='udd'):
        '''
        Obtains the probability that a life x dies survives to age x+t and dies before x+t+n
        :param method: the method used to approximate lx for non-integer x's
        :param x: age at beginning
        :param t: deferment period
        :param n: period
        :return: probability of x dying after age x+t and before x+t+n
        '''
        l_x = self.get_lx_method(x, method)
        l_x_t = self.get_lx_method(x + t, method)
        l_x_t_n = self.get_lx_method(x + t + n, method)
        self.msn.append(f"{t}|{n}_q_{x}={t}_p_{x}  {n}_q_{x + t}={l_x_t} / {l_x} ({l_x_t}-{l_x_t_n}) / {l_x_t}")
        return (l_x_t - l_x_t_n) / l_x

    def force_qw_0(self):
        '''
        forces the last qx to be equal to zero, to state that there are no more decrements after w
        :return: the qx, px and lx adjusted
        '''
        self.__qx[-1] = 0
        self.__px[-1] = 1
        self.__lx[-1] = self.__lx[-2:-1][0]
        self.__dx[-1] = 0

    def exn(self, x, n, method='udd'):
        '''
        Computes the approximated life expectancy between two ages.
        :param x: Initial age.
        :param n: Period to be considered.
        :param method: The method used to interpolate.
        :return: The approximated life span between x and x+n, considering the interpolation methods implemented to approximate the integral of the survival function.
        '''
        if method not in self.__methods:
            return np.nan
        if x < 0:
            return np.nan
        if n <= 0:
            return 1.

        n_max = n
        if x + n > self.w:
            n_max = self.w - x + 1

        if x == int(x):
            next_age = x
        else:
            next_age = int(x) + 1
        to_complete_age = np.round(next_age - x, 6)
        if to_complete_age >= n_max:
            return n_max * self.npx(x, n_max / 2, method)

        if to_complete_age > 0:
            integral1 = to_complete_age * self.npx(x, to_complete_age / 2, method)
        else:
            integral1 = 0

        complete_years = int(n_max - to_complete_age)
        integrals = [self.npx(x, to_complete_age + t, method) *
                     self.get_integral_px_method(next_age + t, method) for t in range(complete_years)]
        final_period = np.round(n_max - to_complete_age - int(n_max - to_complete_age), 6)

        if final_period > 0:
            integral2 = self.npx(x, int(n_max - to_complete_age), method) * \
                        final_period * self.npx(next_age + int(n - to_complete_age), final_period, method)
        else:
            integral2 = 0

        return integral1 + sum(integrals) + integral2
