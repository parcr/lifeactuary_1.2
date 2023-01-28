__author__ = "PedroCR"

import numpy as np
import pandas as pd
from lifeActuary.commutation_table import CommutationFunctions


class CommutationFunctionsFrac(CommutationFunctions):
    """
    Instantiates, for a specific mortality table and interest rate, all the usual commutation functions:
    Dx, Nx, Sx, Cx, Mx, Rx.

    :return: the commutation symbols Dx, Nx, Sx, Cx, Mx, Rx.
    """

    def __init__(self, i=None, g=0, data_type='q', mt=None, perc=100,
                 frac=2, method='udd'):
        CommutationFunctions.__init__(self, i, g, data_type, mt, perc, app_cont=False)
        if method not in self.methods:
            return
        if frac <= 0 or not isinstance(frac, int):
            return

        self.__method = method
        self.__frac = frac

        self.__ages = np.linspace(0, self.w + 1, (self.w + 1) * self.frac + 1)

        radical = 100000.

        self.__lx_frac = np.array([self.npx(x=0, n=x, method=self.__method) for x in self.__ages])
        self.__lx_frac *= radical
        self.__px_frac = np.array([self.npx(x=x, n=1 / self.__frac, method=self.__method) for x in self.__ages])
        self.__qx_frac = 1 - self.__px_frac
        self.__dx_frac = self.__lx_frac[:-1] - self.__lx_frac[1:]
        self.__dx_frac = np.append(self.__dx_frac, 0)

        # Commutations Functions
        self.__Dx_frac = self.__lx_frac[:] * np.power(self.d, self.__ages)
        self.__Nx_frac = np.array([np.sum(self.__Dx_frac[x:]) for x in range(len(self.__ages))])
        self.__Sx_frac = np.array([np.sum(self.__Nx_frac[x:]) for x in range(len(self.__ages))])
        self.__Cx_frac = self.dx_frac * np.power(self.d, self.__ages + 1 / self.__frac)
        self.__Mx_frac = np.array([np.sum(self.__Cx_frac[x:]) for x in range(len(self.__ages))])
        self.__Rx_frac = np.array([np.sum(self.__Mx_frac[x:]) for x in range(len(self.__ages))])

    def __repr__(self):
        return f"{self.__class__.__name__}{self.i, self.g, self.data_type, self.mt, self.perc, self.frac, self.method} "

    def df_commutation_table_frac(self):
        data1 = {'x': self.__ages, 'lx': self.__lx_frac[:], 'dx': self.__dx_frac,
                 'qx': self.__qx_frac, 'px': self.__px_frac}
        data2 = {'Dx': self.__Dx_frac, 'Nx': self.__Nx_frac, 'Sx': self.__Sx_frac, 'Cx': self.__Cx_frac,
                 'Mx': self.__Mx_frac, 'Rx': self.__Rx_frac}
        data = {**data1, **data2}
        df = pd.DataFrame(data)
        return df

    # getters and setters
    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, m):
        self.__method = m

    @property
    def frac(self):
        return self.__frac

    @frac.setter
    def frac(self, f):
        self.__frac = f

    @property
    def ages(self):
        return self.__ages

    @property
    def lx_frac(self):
        return self.__lx_frac

    @property
    def px_frac(self):
        return self.__px_frac

    @property
    def qx_frac(self):
        return self.__qx_frac

    @property
    def dx_frac(self):
        return self.__dx_frac

    @property
    def Dx_frac(self):
        return self.__Dx_frac

    @property
    def Nx_frac(self):
        return self.__Nx_frac

    @property
    def Sx_frac(self):
        return self.__Sx_frac

    @property
    def Cx_frac(self):
        return self.__Cx_frac

    @property
    def Mx_frac(self):
        return self.__Mx_frac

    @property
    def Rx_frac(self):
        return self.__Rx_frac

    def age_to_index(self, age_int, age_frac):
        """
        Allows us to get the index for a specific age and use all the vectors produced by the class to make computations
        using the age .
        """
        parts = np.round(age_frac * self.frac, 5)
        if int(parts) == parts and age_int == int(age_int):
            return int((age_int + age_frac) * self.__frac)
        return np.nan
