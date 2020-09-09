# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from heapq import heappush, heappop, heapify
from copy import deepcopy


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    from collections import deque
    seen = {}
    path = {}
    searchQueue = deque()
    startPoint = maze.getStart()
    searchQueue.append(startPoint)
    path[startPoint] = [startPoint]
    while searchQueue:
        currPoint = searchQueue.popleft()
        if currPoint in seen:
            continue
        else:
            seen[currPoint] = True

            if maze.isObjective(currPoint[0], currPoint[1]):
                return path[currPoint]

            neighbors = maze.getNeighbors(currPoint[0], currPoint[1])
            for n in neighbors:
                if n not in path or len(path[n])-1 > len(path[currPoint]):
                    path[n] = deepcopy(path[currPoint])
                    path[n].append(n)
            searchQueue.extend(neighbors)

    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def dist_to_obj(path_len, point, obj):
        return abs(point[0]-obj[0][0]) + abs(point[1]-obj[0][1]) + path_len

    seen = {}
    path = {}
    objective = maze.getObjectives()
    startPoint = maze.getStart()
    path[startPoint] = [startPoint]
    searchQueue = [(dist_to_obj(0, startPoint, objective), startPoint)]
    heapify(searchQueue)

    while searchQueue:
        _,currPoint = heappop(searchQueue)
        if currPoint in seen:
            continue
        else:
            seen[currPoint] = True

            if maze.isObjective(currPoint[0], currPoint[1]):
                return path[currPoint]

            neighbors = maze.getNeighbors(currPoint[0], currPoint[1])
            for n in neighbors:
                if n not in path or len(path[n]) - 1 > len(path[currPoint]):
                    path[n] = deepcopy(path[currPoint])
                    path[n].append(n)
                    heappush(searchQueue, (dist_to_obj(len(path[n]) - 1, n, objective), n))
                    if n in seen:
                        seen.pop(n)
    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def dist_to_obj(path_len, point, obj):
        return abs(point[0] - obj[0]) + abs(point[1] - obj[1]) + path_len

    def point_to_obj_search(searchQueue, o, seen, path):
        while searchQueue:
            _, currPoint = heappop(searchQueue)
            if currPoint in seen:
                continue
            else:
                seen[currPoint] = True

                if currPoint == o:
                    return path[o]

                neighbors = maze.getNeighbors(currPoint[0], currPoint[1])
                for n in neighbors:
                    if n not in path or len(path[n]) - 1 > len(path[currPoint]):
                        path[n] = deepcopy(path[currPoint])
                        path[n].append(n)
                        heappush(searchQueue, (dist_to_obj(len(path[n]) - 1, n, o), n))
                        if n in seen:
                            seen.pop(n)

    objectives = maze.getObjectives()
    total_path = {o:{} for o in objectives}
    start_to_obj_path = {o:[] for o in objectives}
    origStart = maze.getStart()

    for obj in objectives:
        temp_objs = deepcopy(objectives)
        temp_objs.remove(obj)
        startPoint = obj
        for o in temp_objs:
            # if obj in path[o]:
            #     path[o][obj] = deepcopy(path[obj][o][:-1])
            #     continue
            seen = {}
            path = {obj:[obj]}
            searchQueue = [(dist_to_obj(0, startPoint, o), startPoint)]
            heapify(searchQueue)
            total_path[obj][o] = point_to_obj_search(searchQueue, o, seen, path)
        seen = {}
        path = {origStart:[origStart]}
        searchQueue = [(dist_to_obj(0, startPoint, obj), origStart)]
        heapify(searchQueue)
        start_to_obj_path[obj] = point_to_obj_search(searchQueue, obj, seen, path)

    path_order = []
    for obj in objectives:
        temp_obj0 = deepcopy(objectives)
        temp_obj0.remove(obj)
        for o0 in temp_obj0:
            temp_obj1 = deepcopy(temp_obj0)
            temp_obj1.remove(o0)
            for o1 in temp_obj1:
                temp_obj2 = deepcopy(temp_obj1)
                temp_obj2.remove(o1)
                for o2 in temp_obj2:
                    path_order.append((obj, o0, o1, o2))

    path_combo = []
    for path in path_order:
        path_combo.append(start_to_obj_path[path[0]] + total_path[path[0]][path[1]][1:] + total_path[path[1]][path[2]][1:] +
                          total_path[path[2]][path[3]][1:])
    path_combo.sort(key=len)

    return path_combo[0]

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def manhattan(point, obj):
        return abs(point[0] - obj[0]) + abs(point[1] - obj[1])

    def dist_to_obj(path_len, point, obj):
        STEP_LEN = 15
        row_diff = abs(point[0] - obj[0])
        col_diff = abs(point[1] - obj[1])
        additional_len = 0
        if point[0]>obj[0]:
            for i in range(1,row_diff):
                if maze.isWall(point[0]-i,point[1]):
                    additional_len+=1
        elif point[0]<obj[0]:
            for i in range(1,row_diff):
                if maze.isWall(point[0]+i,point[1]):
                    additional_len+=1

        if point[1]>obj[1]:
            for i in range(1,col_diff):
                if maze.isWall(point[0],point[1]-i):
                    additional_len+=1
        elif point[1]<obj[1]:
            for i in range(1,col_diff):
                if maze.isWall(point[0],point[1]+i):
                    additional_len+=1

        seen = {}
        path = {point: [point]}
        searchQueue = [(manhattan(point, obj), point, obj)]
        heapify(searchQueue)
        for i in range(STEP_LEN):
            dist, currPoint, obj = heappop(searchQueue)
            if currPoint in seen:
                continue
            else:
                seen[currPoint] = True
                if currPoint == obj:
                    break
                neighbors = maze.getNeighbors(currPoint[0], currPoint[1])
                for n in neighbors:
                    if n not in seen or len(path[n]) - 1 > len(path[currPoint]):
                        path[n] = deepcopy(path[currPoint])
                        path[n].append(n)
                        heappush(searchQueue, (len(path[n]) - 1 + manhattan(n, obj), n, obj))
                        if n in seen:
                            seen.pop(n)

        additional_len+=dist

        return row_diff + col_diff + path_len + additional_len

    seen = {}
    path = {}
    objectives = maze.getObjectives()
    startPoint = maze.getStart()
    path[startPoint] = [startPoint]
    searchQueue = [(dist_to_obj(0, startPoint, obj), startPoint, obj) for obj in objectives]
    heapify(searchQueue)

    while objectives:
        _, currPoint, obj = heappop(searchQueue)
        if currPoint in seen:
            continue
        else:
            seen[currPoint] = True
            if currPoint == obj:
                seen.clear()
                objectives.remove(obj)
                searchQueue = [(dist_to_obj(0, currPoint, o), currPoint, o) for o in objectives]
                heapify(searchQueue)
                continue

            neighbors = maze.getNeighbors(currPoint[0], currPoint[1])
            for n in neighbors:
                if n not in seen or len(path[n]) - 1 > len(path[currPoint]):
                    path[n] = deepcopy(path[currPoint])
                    path[n].append(n)
                    heappush(searchQueue, (dist_to_obj(len(path[n]) - 1, n, obj), n, obj))
                    if n in seen:
                        seen.pop(n)


    return path[obj]


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
