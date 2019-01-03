#!/usr/bin/python3
# coding=utf-8

# file containing MOOP and constraints BIOBJ problem
import math


nVar = 2
LB = [0]*nVar
UB = [math.pi]*nVar


def MOOP(solution):
    x = solution.Position
    Cost = []

    objective1 = x[0]
    objective2 = x[1]

    Cost.append(objective1)
    Cost.append(objective2)

    return Cost


def nonlincnstr(solution):
    x = solution.Position
    ceq, c = [], []

    c1 = -(x[0]**2) -(x[1]**2) +1 +0.1*math.cos(16*math.atan(x[0]/x[1]))
    c2 = (x[1]-0.5)**2 + (x[0]-0.5)**2 -0.5

    c.append(c1)
    c.append(c2)

    return ceq, c
