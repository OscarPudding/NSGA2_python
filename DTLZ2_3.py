#!/usr/bin/python3
# coding=utf-8

# file containing MOOP and constraints BIOBJ problem
import math


nVar = 12
LB = [0]*nVar
UB = [1]*nVar


def MOOP(solution):
    x = solution.Position
    Cost = []

    g = 0
    for value in x:
        g = g+(value-0.5)**2

    objective1 = (1+g)*math.cos(x[0]*math.pi/2)*math.cos(x[1]*math.pi/2)
    objective2 = (1+g)*math.cos(x[0]*math.pi/2)*math.sin(x[1]*math.pi/2)
    objective3 = (1+g)*math.sin(x[0]*math.pi/2)

    Cost.append(objective1)
    Cost.append(objective2)
    Cost.append(objective3)

    return Cost


def nonlincnstr(solution):
    x = solution.Position
    ceq, c = [], []

    return ceq, c
