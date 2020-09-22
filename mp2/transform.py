
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """

    ALPHA_INIT, BETA_INIT = arm.getArmAngle()
    ALPHA_LIM, BETA_LIM = arm.getArmLimit()
    rows = int((ALPHA_LIM[1]-ALPHA_LIM[0])/granularity) + 1
    cols = int((BETA_LIM[1]-BETA_LIM[0])/granularity) + 1
    input_map = [[SPACE_CHAR for c in range(cols)] for r in range(rows)]
    for alpha in range(ALPHA_LIM[0], ALPHA_LIM[1]+1, granularity):
        for beta in range(BETA_LIM[0], BETA_LIM[1]+1, granularity):
            arm.setArmAngle((alpha,beta))
            armPosDist = arm.getArmPosDist()
            armTip = arm.getEnd()
            armPos = arm.getArmPos()
            if doesArmTipTouchGoals(armTip, goals) and not doesArmTouchObjects(armPosDist, obstacles) and \
                    isArmWithinWindow(armPos, window):
                row, col = angleToIdx([alpha, beta], [ALPHA_LIM[0], BETA_LIM[0]], granularity)
                input_map[row][col] = OBJECTIVE_CHAR
            else:
                if doesArmTouchObjects(armPosDist, obstacles) or not isArmWithinWindow(armPos, window):
                    row, col = angleToIdx([alpha, beta], [ALPHA_LIM[0], BETA_LIM[0]], granularity)
                    input_map[row][col] = WALL_CHAR

    row, col = angleToIdx([ALPHA_INIT, BETA_INIT], [ALPHA_LIM[0], BETA_LIM[0]], granularity)
    input_map[row][col] = START_CHAR

    return Maze(input_map, [ALPHA_LIM[0], BETA_LIM[0]], granularity)