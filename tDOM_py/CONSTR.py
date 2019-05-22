#!/usr/bin/python3
# coding=utf-8

# file containing MOOP and constraints BIOBJ problem
import math


nVar = 2
LB = [0.1, 0.0]
UB = [1.0, 5.0]


def MOOP(solution):
    x = solution.Position
    Cost = []

    objective1 = x[0]
    objective2 = (1+x[1])/x[0]

    Cost.append(objective1)
    Cost.append(objective2)

    return Cost


def nonlincnstr(solution):
    x = solution.Position
    ceq, c = [], []

    c1 = -x[1]-9*x[0]+6
    c2 = x[1]-9*x[0]+1

    c.append(c1)
    c.append(c2)

    return ceq, c


def SOOP(solution, iteration):
    x = solution
    objective1 = x[0]
    objective2 = (1 + x[1]) / x[0]

    if iteration[0]==0:
        return objective1
    else:
        return objective2


def cons(solution):
    x = solution
    c = []
    c1 = -x[1] - 9 * x[0] + 6
    c2 = x[1] - 9 * x[0] + 1

    c.append(c1)
    c.append(c2)

    return c
