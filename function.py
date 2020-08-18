import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import brentq
from math import exp



###lambda1
def Equil1(spread, lmd1, DF, R):
    prel = spread*sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))
    prol = (1-R)*sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))
    return prel-prol




###lambda2
def Equil2(spread, lmd1, lmd2, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5)))
    return prel-prol


###lambda3
def Equil3(spread, lmd1, lmd2, lmd3, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9)))
    return prel-prol



###lambda4
def Equil4(spread, lmd1, lmd2, lmd3, lmd4, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13)))
    return prel-prol



###lambda5
def Equil5(spread, lmd1, lmd2, lmd3, lmd4, lmd5, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17)))
    return prel-prol



###lambda6
def Equil6(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21)))
    return prel-prol
###lambda7
def Equil7(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, lmd7, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7))*
                       DF[n-1] for n in range(21, 29)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-21)*lmd7))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7)))*
                      DF[n-1] for n in range(21, 29)))
    return prel-prol

###lambda8
def Equil8(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, lmd7, lmd8, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7))*
                       DF[n-1] for n in range(21, 29))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8))*
                       DF[n-1] for n in range(29, 41)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-21)*lmd7))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7)))*
                      DF[n-1] for n in range(21, 29))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-29)*lmd8))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8)))*
                      DF[n-1] for n in range(29, 41)))
    return prel-prol

###lambda9
def Equil9(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, lmd7, lmd8, lmd9, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7))*
                       DF[n-1] for n in range(21, 29))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8))*
                       DF[n-1] for n in range(29, 41))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+
                                          8*lmd7+12*lmd8+(n-40)*lmd9))*DF[n-1] for n in range(41, 61)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-21)*lmd7))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7)))*
                      DF[n-1] for n in range(21, 29))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-29)*lmd8))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8)))*
                      DF[n-1] for n in range(29, 41))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-41)*lmd9))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-40)*lmd9)))*
                      DF[n-1] for n in range(41, 61)))
    return prel-prol


###lambda10
def Equil10(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, lmd7, lmd8, lmd9, lmd10, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7))*
                       DF[n-1] for n in range(21, 29))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8))*
                       DF[n-1] for n in range(29, 41))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+
                                          8*lmd7+12*lmd8+(n-40)*lmd9))*DF[n-1] for n in range(41, 61))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                          12*lmd8+20*lmd9+(n-60)*lmd10))*DF[n-1] for n in range(61, 81)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-21)*lmd7))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7)))*
                      DF[n-1] for n in range(21, 29))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-29)*lmd8))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8)))*
                      DF[n-1] for n in range(29, 41))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-41)*lmd9))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-40)*lmd9)))*
                      DF[n-1] for n in range(41, 61))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                     12*lmd8+20*lmd9+(n-61)*lmd10))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                     12*lmd8+20*lmd9+(n-60)*lmd10)))*
                      DF[n-1] for n in range(61, 81)))
    return prel-prol

###lambda11
def Equil11(spread, lmd1, lmd2, lmd3, lmd4, lmd5, lmd6, lmd7, lmd8, lmd9, lmd10, lmd11, DF, R):
    prel = spread*(sum(0.25*np.exp(-0.25*n*lmd1)*DF[n-1] for n in range(1, 3))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+(n-2)*lmd2))*DF[n-1] for n in range(3, 5))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3))*DF[n-1] for n in range(5, 9))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4))*DF[n-1] for n in range(9, 13))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5))*
                       DF[n-1] for n in range(13, 17))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6))*
                       DF[n-1] for n in range(17, 21))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7))*
                       DF[n-1] for n in range(21, 29))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8))*
                       DF[n-1] for n in range(29, 41))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+
                                          8*lmd7+12*lmd8+(n-40)*lmd9))*DF[n-1] for n in range(41, 61))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                          12*lmd8+20*lmd9+(n-60)*lmd10))*DF[n-1] for n in range(61, 81))+
                   sum(0.25*np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+
                                          8*lmd7+12*lmd8+20*lmd9+20*lmd10+
                                          (n-80)*lmd11))*DF[n-1] for n in range(81, 121)))
    prol = (1-R)*(sum((np.exp(-0.25*(n-1)*lmd1)-np.exp(-0.25*n*lmd1))*DF[n-1] for n in range(1, 3))+
                  sum((np.exp(-0.25*(2*lmd1+(n-3)*lmd2))-
                       np.exp(-0.25*(2*lmd1+(n-2)*lmd2)))*DF[n-1] for n in range(3, 5))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+(n-5)*lmd3))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+(n-4)*lmd3)))*DF[n-1] for n in range(5, 9))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-9)*lmd4))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+(n-8)*lmd4)))*DF[n-1] for n in range(9, 13))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-13)*lmd5))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+(n-12)*lmd5)))*DF[n-1] for n in range(13, 17))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-17)*lmd6))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+(n-16)*lmd6)))*
                      DF[n-1] for n in range(17, 21))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-21)*lmd7))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+(n-20)*lmd7)))*
                      DF[n-1] for n in range(21, 29))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-29)*lmd8))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+(n-28)*lmd8)))*
                      DF[n-1] for n in range(29, 41))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-41)*lmd9))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+(n-40)*lmd9)))*
                      DF[n-1] for n in range(41, 61))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                     12*lmd8+20*lmd9+(n-61)*lmd10))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+
                                     12*lmd8+20*lmd9+(n-60)*lmd10)))*
                      DF[n-1] for n in range(61, 81))+
                  sum((np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+
                                     20*lmd9+20*lmd10+(n-81)*lmd11))-
                       np.exp(-0.25*(2*lmd1+2*lmd2+4*lmd3+4*lmd4+4*lmd5+4*lmd6+8*lmd7+12*lmd8+
                                     20*lmd9+20*lmd10+(n-80)*lmd11)))*
                      DF[n-1] for n in range(81, 121)))
    return prel-prol
