#!/usr/bin/python3
# coding=utf-8

# file containing MOOP and constraints BIOBJ problem

nVar = 2
LB = [-10]*nVar
UB = [10]*nVar


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

    c1 = ((x[0] - 10) / 10) ** 8 + ((x[1] - 5) / 5) ** 8 - 1
    c.append(c1)

    return ceq, c
