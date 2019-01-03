#!/usr/bin/python3
# coding=utf-8

# file containing MOOP and constraints BIOBJ problem
import math


nVar = 300
LB = [0]*nVar
UB = [1]*nVar


def MOOP(solution):
    x = solution.Position
    Cost = []

    n = 300
    s = 1.0
    k = 4.0
    g1 = lambda z: 1 + (9 / (n + 1)) * sum(z[1:])
    r = lambda z: 5 + 10 * (z - 0.5) ** 2 + (1 / k) * math.cos(2 * k * math.pi * z) * (2 ** (s / 2))
    objective1 = g1(x) * r(x[0]) * math.sin(
        math.pi * (x[0] / (2 ** (s + 1))) + (1 + ((2 ** s - 1) / (2 ** (s + 2))) * math.pi) + 1)
    objective2 = g1(x) * r(x[0]) * (math.cos((math.pi * x[0] / 2) + math.pi) + 1)

    Cost.append(objective1)
    Cost.append(objective2)

    return Cost


def nonlincnstr(solution):
    x = solution.Position
    ceq, c = [], []

    return ceq, c
