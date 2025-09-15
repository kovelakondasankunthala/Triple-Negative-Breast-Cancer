import numpy as np
import time

# Position update is done at line 53
def MMTBO(pop_Position, fobj, lb, ub, Max_iteration):
    #  'MTBO: Mountaineering Team Based Optimization'
    [N, dim] = pop_Position.shape
    nVar = dim
    VarSize = np.array([N, dim])
    VarMin = lb
    VarMax = ub
    MaxIt = Max_iteration
    nPop = N
    empty_individual_Position = []
    empty_individual_Cost = []
    BestSol_Cost = np.inf
    Convergence_curve = np.zeros((Max_iteration, 1))
    newsol_Cost = np.zeros(nPop)
    pop_Cost = np.zeros(nPop)
    # Initialize Population Members
    for i in range(nPop):
        # pop(i).Position = np.random.uniform(VarMin, VarMax, (VarSize))
        # pop_Position[:] = np.random.uniform(np.min(VarMin), np.max(VarMax), (VarSize))
        pop_Cost[i] = fobj(pop_Position[i, :])

        if pop_Cost[i] < BestSol_Cost:
            BestSol = pop_Position[i]

    BestCosts = np.zeros((MaxIt, 1))
    ct = time.time()
    ## MTBO Main Loop
    for it in range(MaxIt):
        # Calculate Population Mean
        Mean = 0
        for i in range(nPop):
            Mean = Mean + pop_Position[i]
        Mean = Mean / nPop
        # Select Leader
        Leader_Cost = np.min(pop_Position)
        for i in range(nPop):
            if pop_Cost[i] < Leader_Cost:
                Leader_Position = pop_Position[i]
        for i in range(nPop):
            # Create Empty Solution
            newsol_Position = []
            ii = i + 1
            if ii > nPop:
                ii = 1
            Li = (0.25 + 0.25 * np.random.rand())
            Ai = (0.75 + 0.25 * np.random.rand())
            Mi = (0.75 + 0.25 * np.random.rand())
            # position is update using this r
            r = -it * ((-1) / MaxIt)
            if r < Li:
                newsol_Position = pop_Position[i] + np.multiply(np.random.rand(N, dim),
                                                                (pop_Position[i] - pop_Position[i])) + np.multiply(
                    np.random.rand(N, dim), (Leader_Position - pop_Position[i]))
            else:
                if np.random.rand() < Ai:
                    newsol_Position = pop_Position[i] + np.multiply(1 * np.random.rand(N, dim),
                                                                    (pop_Position[i] - pop_Position[nPop - 1]))
                else:
                    if np.random.rand() < Mi:
                        newsol_Position = pop_Position[i] + np.multiply(1 * np.random.rand(N, dim),
                                                                        (Mean - pop_Position[i]))
                    else:
                        newsol_Position = np.random.uniform(VarMin, VarMax, (N, dim))
            newsol_Cost[i] = fobj(newsol_Position[i])
            if newsol_Cost[i] < pop_Cost[i]:
                pop_Position[i] = newsol_Position[i]
                if pop_Cost[i] < Leader_Cost:
                    Leader_Cost = min(pop_Position[i])
        uuu, SortOrder = np.sort(pop_Cost), np.argsort(pop_Cost)
        pop = pop_Position[SortOrder]
        Convergence_curve[it] = np.min(pop_Cost)

    Destination_fitness = np.min(pop_Cost)
    Destination_position = pop_Position[0]
    ct += time.time()

    return Destination_fitness, Convergence_curve, Destination_position, ct
