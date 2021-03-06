#!/usr/bin/python3
# coding=utf-8

import random
import winsound
import math
import operator
import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from TNK import MOOP, nonlincnstr, nVar, LB, UB

# General import structure
# from CASE_STUDY import MOOP, nonlincnstr


class Solution:
    def __init__(self):
        self.Position = []
        self.Cost = []
        self.Rank = 0
        self.DominationSet = []
        self.DominatedCount = 0
        self.CrowdingInWaiting = []
        self.CrowdingDistance = 0


def nondomsorting(pop_c):
    # Non-dominated sorting step
    npop = len(pop_c)
    for i1 in pop_c:
        i1.DominationSet = []
        i1.DominatedCount = 0

    front1 = []
    frontier = []
    for i2 in range(0, npop):
        for i3 in range(i2+1, npop):
            p = pop_c[i2]
            q = pop_c[i3]
            if dominates(p, q):
                p.DominationSet.append(i3)
                q.DominatedCount = q.DominatedCount +1

            if dominates(q, p):
                q.DominationSet.append(i2)
                p.DominatedCount = p.DominatedCount +1

            pop_c[i2] = p
            pop_c[i3] = q

        if pop_c[i2].DominatedCount == 0:
            front1.append(i2)
            pop_c[i2].Rank = 1

    ki = 0
    frontier.append(front1)

    while True:
        Q = []
        for index1 in frontier[ki]:
            p = pop_c[index1]
            for index2 in p.DominationSet:
                q = pop_c[index2]
                q.DominatedCount = q.DominatedCount - 1
                if q.DominatedCount == 0:
                    Q.append(index2)
                    q.Rank = ki + 2

                pop_c[index2] = q

        if not Q:
            break

        frontier.append(Q)
        ki = ki +1

    return pop_c, frontier


def dominates(x, y):
    # x dominates y if and only if all elements in <= than y and at least one <
    leq = []
    for idom1 in range(0, len(x.Cost)):
        if x.Cost[idom1] <= y.Cost[idom1]:
            leq.append(1)
        else:
            leq.append(0)

    if all(leq):
        leq2 = []
        for idom2 in range(0, len(x.Cost)):
            if x.Cost[idom2] < y.Cost[idom2]:
                leq2.append(1)
            else:
                leq2.append(0)

        if any(leq2):
            return True
        else:
            return False
    else:
        return False


def crowdingdist(pop_crow, fronts):
    # Crowding distance calculation
    nobj = len(pop_crow[1].Cost)
    for icd1 in range(0,len(fronts)):
        for obj in range(0, nobj):
            Costsobj = []
            for ind in fronts[icd1]:
                Costsobj.append(pop_crow[ind].Cost[obj])

            for it1 in range(0, len(Costsobj)):
                Costsobj[it1] = [Costsobj[it1], fronts[icd1][it1]]

            Costsobj.sort(key=lambda x: x[0])

            for it2 in range(1, len(Costsobj)-1):
                if len(Costsobj) > 1:
                    ranger = abs(Costsobj[-1][0]-Costsobj[0][0])
                    if ranger == 0:
                        pop_crow[Costsobj[it2][1]].CrowdingInWaiting.append(0)
                    else:
                        crowding = (Costsobj[it2+1][0] - Costsobj[it2-1][0])/ranger
                        pop_crow[Costsobj[it2][1]].CrowdingInWaiting.append(crowding)

                else:
                    for it3 in range(0, len(Costsobj)):
                        pop_crow[Costsobj[it3][1]].CrowdingDistance = math.inf

            pop_crow[Costsobj[0][1]].CrowdingDistance = math.inf
            pop_crow[Costsobj[-1][1]].CrowdingDistance = math.inf

        for ind1 in fronts[icd1]:
            crow = pop_crow[ind1].CrowdingInWaiting
            pop_crow[ind1].CrowdingDistance = np.mean(crow)

    return pop_crow


def mutation(x):
    # Offspring 1: Mutation step
    numvar = 0.5  # fraction of variables that are mutated
    stepsize = 0.2  # % how much will they be mutated
    muta = Solution()
    muta.Position = x.Position
    index = []
    for im1 in range(0, math.ceil(len(muta.Position) * numvar)):
        index.append(random.randint(0, len(muta.Position) - 1))

    for im2 in index:
        a = random.uniform(0, 1)
        if a < 0.5:
            muta.Position[im2] = muta.Position[im2] - stepsize * muta.Position[im2]
        else:
            muta.Position[im2] = muta.Position[im2] + stepsize * muta.Position[im2]

    for i in range(0, len(muta.Position)):
        if muta.Position[i] < LB[i]:
            muta.Position[i] = random.uniform(LB[i],UB[i])
        elif muta.Position[i] > UB[i]:
            muta.Position[i] = random.uniform(LB[i],UB[i])

    return muta


def crossover(x, y):
    # Offspring 2: Crossover step
    pos1 = x.Position
    pos2 = y.Position
    howmany = 1
    if len(pos1)-1 > howmany:
        howmany = random.randrange(1, len(pos1)-1)

    index = []
    for i in range(0, howmany):
        index.append(random.randrange(0, len(pos1)))

    x_cross = Solution()
    y_cross = Solution()
    x_cross.Position = pos1
    y_cross.Position = pos2
    for ic in index:
        x_cross.Position[ic], y_cross.Position[ic] = y_cross.Position[ic], x_cross.Position[ic]

    for i in range(0, len(x_cross.Position)):
        if x_cross.Position[i] < LB[i]:
            x_cross.Position[i] = random.uniform(LB[i],UB[i])
        elif x_cross.Position[i] > UB[i]:
            x_cross.Position[i] = random.uniform(LB[i],UB[i])

        if y_cross.Position[i] < LB[i]:
            y_cross.Position[i] = random.uniform(LB[i],UB[i])
        elif y_cross.Position[i] > UB[i]:
            y_cross.Position[i] = random.uniform(LB[i],UB[i])

    return x_cross, y_cross


def sorting(pop_select):
    # Sorting the solutions in pop_select according to their fitness
    # Sorting according to crowding distance
    sorter1 = []
    pop_sorted1 = []
    for sol in range(0, len(pop_select)):
        sorter1.append([pop_select[sol].CrowdingDistance, sol])

    sorter1 = sorted(sorter1, key=operator.itemgetter(0), reverse=True)
    for itsort1 in range(0, len(sorter1)):
        pop_sorted1.append(pop_select[sorter1[itsort1][1]])

    # Sorting according to non-dominated rank
    sorter2 = []
    pop_sorted = []
    for sol2 in range(0, len(pop_sorted1)):
        sorter2.append([pop_sorted1[sol2].Rank, sol2])

    sorter2 = sorted(sorter2, key=operator.itemgetter(0))
    for itsort2 in range(0, len(sorter2)):
        pop_sorted.append(pop_sorted1[sorter2[itsort2][1]])

    return pop_sorted


def infeasible(opl):
    # BIOBJ
    # c1 = ((opl.Position[0]-10)/10)**8 + ((opl.Position[1]-5)/5)**8 - 1
    ceq, c = nonlincnstr(opl)
    if any(opl1 != 0 for opl1 in ceq) or any(opl2 > 0 for opl2 in c):
        return True
    else:
        return False

    # DO2DK
    # c1 = -1  # no inequality constraints

    #if c1 > 0:
        # return True  # The solution is infeasible
    # else:
        # return False  # The solution is feasible


def mainloop(pops, npop, pc, pm):
    pop_cross = []
    pop_mut = []
    # Crossover step
    ncross = math.ceil((npop * pc) / 2)
    for j in range(0, ncross):
        cross1, cross2 = crossover(pops[random.randrange(0, nPop)], pops[random.randrange(0, nPop)])
        cross1.Cost = MOOP(cross1)
        cross2.Cost = MOOP(cross2)
        counter2 = 0
        while infeasible(cross1):
            cross1.Position = []
            cross1.Cost = []
            cross1.Position = randomposition(nVar)
            cross1.Cost = MOOP(cross1)

        counter3 = 0
        while infeasible(cross2):
            cross2.Position = []
            cross2.Cost = []
            cross2.Position = randomposition(nVar)
            cross2.Cost = MOOP(cross2)

        pop_cross.append(cross1)
        pop_cross.append(cross2)

    # Mutation step
    nmut = math.ceil(npop * pm)
    for j in range(0, nmut):
        mut = mutation(pops[random.randrange(0, nPop)])
        mut.Cost = MOOP(mut)
        counter = 0
        while infeasible(mut):
            mut.Position = []
            mut.Cost = []
            mut.Position = randomposition(nVar)
            mut.Cost = MOOP(mut)

        pop_mut.append(mut)

    # Merging
    popper = []
    popper.extend(pops)
    popper.extend(pop_cross)
    popper.extend(pop_mut)

    # Non-dominated sorting step
    pop_nd, front = nondomsorting(popper)

    # Crowding distance calculation
    pop_crowd = crowdingdist(pop_nd, front)

    # Sorting step
    pop_sortd = sorting(pop_crowd)

    # Selecting step
    popcropped = pop_sortd[0:nPop]

    # Updating non-dominated fronts
    popcroppd, frontsupdated = nondomsorting(popcropped)

    return popcroppd


def randomposition(nvar):
    position = []
    for randi in range(0, nvar):
        position.append(random.uniform(LB[randi], UB[randi]))

    return position


if __name__ == '__main__':
    nPop = 75
    MaxIt = 50
    pmut = 0.10
    pcross = 0.90
    pop = []
    pop_final = []

    # Initialisation
    for i in range(0, nPop):
        pop.append(Solution())
        pop[i].Position = randomposition(nVar)
        pop[i].Cost = MOOP(pop[i])
        # pop[i].objectiveMOOP()
        while infeasible(pop[i]):
            pop[i].Position = []
            pop[i].Cost = []
            pop[i].Position = randomposition(nVar)
            pop[i].Cost = MOOP(pop[i])
            # pop[i].objectiveMOOP()

    for it in range(0, MaxIt):
        print('Iteration {}'.format(it+1))
        popcropp = mainloop(pop, nPop, pcross, pmut)
        pop = popcropp
        if it == MaxIt-1:
            pop_final = popcropp

    fig = mpl.figure(1)
    #ax = fig.add_subplot(111, projection='3d')

    number = 0
    for i in range(0, nPop):
        if pop_final[i].Rank == 1:
            mpl.plot(pop_final[i].Cost[0], pop_final[i].Cost[1], 'b.', markersize=5)
            # ax.scatter(pop_final[i].Cost[0], pop_final[i].Cost[1], pop_final[i].Cost[2], c='b', marker='o')
            number = number + 1

    print('\n{} non-dominated solutions'.format(number))
    # mpl.axis([-0.5, 10, 0, 10])
    # mpl.axis([-1, 45, -1, 45])
    mpl.xlabel("$J_1$")
    mpl.ylabel("$J_2$")

    # for angle in [0, 30, 60, 90, 120]:
        # ax.view_init(30, angle)
        # mpl.draw()
        # mpl.pause(1)

    mpl.show()
