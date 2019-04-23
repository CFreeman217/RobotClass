#!/usr/bin/env python

## module rrt_search
import os, csv, math, datetime, random, time
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# Lower left and upper right coordinate pairs for map boundaries.
map_size = [(0.0,0.0),(20.0,20.0)]
# Position of starting location
start_pos = (0.0,0.0)
# Position of goal location
goal_pos = (19.0,1.0)

def get_obs(bounds, num, goal):
    '''
    Generates a random map of obstacles for the map provided.

    Inputs:
    bounds: Lower left and upper right coordinate pairs [(xmin, ymin), (xmax, ymax)] as tuples
        map boundaries
    num: (integer) number of obstacles to generate
    goal: (xgoal, ygoal) goal location that cannot have an obstacle over it
    '''
    obs_list = []
    boundMin_x = bounds[0][0]
    boundMin_y = bounds[0][1]
    boundMax_x = bounds[1][0]
    boundMax_y = bounds[1][1]
    for i in range(num):
        genx = float(random.randint(boundMin_x, boundMax_x))
        geny = float(random.randint(boundMin_y, boundMax_y))
        while (genx, geny) in obs_list:
            genx = float(random.randint(boundMin_x, boundMax_x))
            geny = float(random.randint(boundMin_y, boundMax_y))
        if (genx, geny) == goal:
            continue
        else:
            obs_list.append((genx, geny))
    return obs_list

# obstacles = get_obs(map_size, 35, goal_pos)

obstacles = [(1.0,1.0),
             (4.0,4.0),
             (3.0,4.0),
             (5.0,0.0),
             (5.0,1.0),
             (0.0,7.0),
             (1.0,7.0),
             (2.0,7.0),
             (3.0,7.0),
             (2.0,10.0),
             (3.0,10.0),
             (4.0,10.0),
             (5.0,10.0),
             (6.0,10.0),
             (7.0,10.0),
             (8.0,10.0),
             (8.0,9.0),
             (8.0,8.0),
             (8.0,7.0),
             (8.0,6.0),
             (8.0,5.0),
             (8.0,4.0),
             (8.0,3.0),
             (8.0,2.0),
             (8.0,1.0),
             (8.0,0.0),]

dist = lambda a, b, c, d : ((abs(c - a)**2 + abs(d - b)**2)**0.5)

def get_filename(prefix, suffix, base_path):
    '''
    Gets a unique file name in the base path.
    
    Appends date and time information to file name and adds a number
    if the file name is stil not unique.

    prefix = Homework assignment name

    suffix = Extension

    base_path = Location of log file
    '''
    # Set base filename for compare
    fileNameBase = base_path + prefix + "_" + datetime.datetime.now().strftime("%b_%d_%H_%M")
    # Set base for numbering system if filename exists
    num = 1
    # Generate complete filename to check existence
    fileName = fileNameBase + suffix
    # Find a unique filename
    while os.path.isfile(fileName):
        # if the filename is not unique, add a number to the end of it
        fileName = fileNameBase + "_" + str(num) + suffix
        # increments the number in case the filename is still not unique
        num = num + 1
    return fileName

class PATHFIND:
    '''
    Pathfinding Algorithms

    Rapidly Expanding Random Tree Algorithm:

    1.) Choose a random location on the map.
    2.) Find the nearest node to this location. This is the parent
        node for the next node to be created.
    3.) Create a new node between nearest node and random loc.
    4.) If new node is within 'L' of goal, trace parent node path
        back to starting position.
    
    Needs map size, start, end, and obstacles.
    - Obstacles will need to be configured with a physical area for
      this to work. See the class method for obstacle creation for
      more information.
    
    Usage:

    map_size: List of extreme coordinates for map boundaries
    obstacles: List of (x,y) coordinate pairs of obstacle locations
    size: Float for size of hitbox for determining permissible path.
          Works best if the step size is smaller than this size.
    goal_pos, start_pos: (x,y) coordinate tuple float for initial
                         and final position
    stp_size: Step size (float) to determine how far to jump each time.

    example:
    map_size = [(lower_left_x_coord, lower_left_y_coord),
                (upper_rght_x_coord, upper_rght_y_coord)]
    obstacles = [(obs1_x, obs1_y), (obs2_x, obs2_y),...,(obsN_x, obsN_y)]
    goal_pos = (goal_loc_x, goal_loc_y)
    start_pos = (start_loc_x, start_loc_y)

    PATHFIND.set_map(map_size, block_size)
    PATHFIND.load_obstacles(obstacles, obs_size)
    PATHFIND.set_goal(goal_pos)
    PATHFIND.set_start(start_pos)
    PATHFIND.run_rrt(stp_size)
    '''
    num_nodes = 0
    map_nodes = {}
    map_obs = []
    start_loc = None
    end_goal = None
    pathway = []
    best_path = []
    

    def __init__(self, x_coord, y_coord, parent=None, start=False):
        '''
        Instantiates an instance of the RRT Node class from a given (x,y) coordinate pair.
        Adds the instance to the class node dictionary if the passed point is valid and not an obstacle.
        '''
        if self.valid_check(x_coord, y_coord):
            PATHFIND.num_nodes += 1
            self.xloc = x_coord
            self.yloc = y_coord
            self.coordinates = (x_coord, y_coord)
            self.cost = 0.0
            self.parent = parent
            self.start_node = start
            self.node_index = PATHFIND.num_nodes
            PATHFIND.map_nodes[PATHFIND.num_nodes] = self
        else:
            print '__init__ : new node creation attempt for invalid point at ({:1.2f},{:1.2f})'.format(x_coord, y_coord)
        
    def distance(self, other):
        '''
        Calculates the distance between nodes. Returns float.
        Usage:
        distance = node_1.distance((x_coord, y_coord))
        '''
        a = self.xloc
        b = self.yloc
        c = other[0]
        d = other[1]
        dist = ((abs(c - a)**2 + abs(d - b)**2)**0.5)
        ang = math.atan2((d-b), (c-a))
        return (dist, ang)

    def valid_check(self, x_val, y_val):
        '''
        Checks to see if the given coordinates are valid for the map boundaries and obstacle layout.
        Pass an (x,y) coordinate pair and will return True if the point can be created.
        '''
        if PATHFIND.x_min <= x_val <= PATHFIND.x_max and PATHFIND.y_min <= y_val <= PATHFIND.y_max:
            for obs in PATHFIND.map_obs:
                if obs[0][0] <= x_val <= obs[1][0] and obs[0][1] <= y_val <= obs[1][1]:
                    print 'valid_check : ({:1.2f},{:1.2f}) collision with obstacle at [({:1.2f},{:1.2f}),({:1.2f},{:1.2f})]'.format(x_val, y_val, obs[0][0], obs[0][1], obs[1][0], obs[1][1])
                    return False
            return True
        else:
            print 'valid_check : ({:1.2f},{:1.2f}) outside map bounds'.format(x_val, y_val)
            return False

    @classmethod
    def set_map(cls, coords, blk_size=1.0):
        '''
        Sets limits on the boundaries for the map and stores the values in class variables to be called
        when validity checking or drawing the setup.
        blk_size = size of blocks to use in path finding
        '''
        half = blk_size/2.0
        cls.map_limits = coords
        cls.x_min = coords[0][0] - half
        cls.x_max = coords[1][0] + half
        cls.y_min = coords[0][1] - half
        cls.y_max = coords[1][1] + half

    @classmethod
    def load_obstacles(cls, obstacles, size):
        '''
        Loads obstacles within the map structure and stores them within the class variable map_obs.
        Size determines the length of the hitbox for an obstacle. (draws square with side length 'size')
        '''
        halfsize = size/2.0
        for obs in obstacles:
            obs_xmin = obs[0] - halfsize
            obs_xmax = obs_xmin + size
            obs_ymin = obs[1] - halfsize
            obs_ymax = obs_ymin + size
            cls.map_obs = cls.map_obs + [((obs_xmin, obs_ymin),(obs_xmax, obs_ymax))]
    
    @classmethod
    def set_goal(cls, coords):
        '''
        This scheme does not need to have an established node at the end location,
        the end goal locaiton just needs to be stored in the class.
        '''
        cls.end_goal = coords

    @classmethod
    def set_start(cls, coords):
        '''
        Sets start position and establishes a node at the origin.
        '''
        cls.start_loc = coords
        cls(coords[0],coords[1], start=True)

    @classmethod
    def draw_map(cls, title='Untitled'):
        '''
        Draws border, obstacles, start node, end node, and intermediate nodes onto the plotting space.
        Depends on the draw_border(), draw_obstacles(), and draw_circ() functions.
        '''
        cls.fig, cls.ax = plt.subplots()
        cls.draw_border()
        cls.draw_obstacles()
        # Start position
        cls.draw_circ(cls.start_loc, 0.2, '#336633')
        # End position
        cls.draw_circ(cls.end_goal, 0.2, '#CC0000')
        for node in cls.map_nodes.values():
            plt.scatter(node.xloc, node.yloc, color='#000000')

        for key, point in enumerate(cls.best_path[:-1]):
            lin_x = [point[0], cls.best_path[key+1][0]]
            lin_y = [point[1], cls.best_path[key+1][1]]
            plt.plot(lin_x, lin_y, color='b')
        cls.ax.set_ylabel('y location', fontsize=12)
        cls.ax.set_xlabel('x location', fontsize=12)
        cls.ax.set_title(title + ' {} Nodes Generated.'.format(cls.num_nodes))
        plt.axis('equal')
        plt.axis([cls.x_min - 1.0,cls.x_max + 1.0,cls.y_min - 1.0,cls.y_max + 1.0])
        plt.tight_layout()


    @classmethod
    def draw_border(cls):
        '''
        Creates a border that outlines the total valid points in the map space
        - This is a sub-function called by the draw_map() function
        '''
        box_color = '#555555'
        ln_width = 3
        cls.ax.plot([cls.x_min, cls.x_max],[cls.y_min, cls.y_min], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_min, cls.x_max],[cls.y_max, cls.y_max], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_min, cls.x_min],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_max, cls.x_max],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)

    @classmethod
    def draw_obstacles(cls):
        '''
        Draws the obstacle blocks in the plotting space
        - This is a sub-function called by the draw_map() function
        '''
        boxes = []
        face_color = '#555555'
        for block in cls.map_obs:
            bxmin = block[0][0]
            bymin = block[0][1]
            bxmax = block[1][0]
            bymax = block[1][1]
            bwidth = bxmax - bxmin
            bheight = bymax - bymin
            rect = mpatches.Rectangle((bxmin, bymin), bwidth, bheight)
            boxes.append(rect)
        pc = PatchCollection(boxes, facecolor=face_color, edgecolor='None')
        cls.ax.add_collection(pc)
    
    @classmethod
    def draw_circ(cls, coords, rad=0.1, cir_color='b'):
        '''
        Adds a circle to the plotting space
        - This is a sub-function called by the new_node() function
        '''
        circ = [mpatches.Circle(coords, radius=rad)]
        pc = PatchCollection(circ, facecolor=cir_color, edgecolor='None')
        cls.ax.add_collection(pc)

    @classmethod
    def run_RRT(cls, step_size, animate=False):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~  RUN_RRT CALLED~~~~~~~~~~~~~~~~~~~~~~'
        # Set plot space header while we are trying to find this path
        cls.draw_map('Processing RRT...')
        while cls.nearest_node(cls.end_goal)[1][0] > step_size:
            # Run this loop while the nearest node to the goal node is larger than the step size
            # Generate a random point
            rand_point = (random.uniform(cls.x_min, cls.x_max), random.uniform(cls.y_min, cls.y_max))
            # Find the nearest node to this generated point
            # Yields node index, distance and angle(radians)
            near_node = cls.nearest_node(rand_point)
            # Generates a new node (if valid) at the new estimated location
            cls.new_node(near_node, step_size, animate)
        # Get the node number of the nearest node to the end goal
        current_node_index = cls.nearest_node(cls.end_goal)[0]
        # Create a node object from this nearest node
        current_node = cls.map_nodes[current_node_index]
        # The goal here is to build a list of node locations that make up our path to the goal
        # The first value should be the end node, or else plotting looks weird
        # Also, the robot which follows this path will stop one node short of the goal without this.
        cls.best_path.append(cls.end_goal)
        # Now we are going to work backwards from the goal node to the start
        # This uses the parent nodes for each node to append to the list
        while current_node.start_node == False:
            cls.best_path.append(current_node.coordinates)
            current_node = cls.map_nodes[current_node.parent]
        # We still need to add the start node coordinates so the plotting system works.
        # This is also to get the robot to find the first generated node from the start.
        cls.best_path.append(current_node.coordinates)
        # print cls.best_path
        # plt.show()
        return cls.best_path[::-1]
    
    @classmethod
    def optimize1(cls, step_size, animate=False):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~  Optimized Algorithm Attempt 1~~~~~~~~~~~~~~~~~~~~~~'
        cls.draw_map('Processing Optimized Attempt 1')
        while cls.nearest_node(cls.end_goal)[1][0] > step_size:
            # INSERT ALGORITHM HERE
            pass
            
        current_node_index = cls.nearest_node(cls.end_goal)[0]
        current_node = cls.map_nodes[current_node_index]
        cls.best_path.append(cls.end_goal)
        while current_node.start_node == False:
            cls.best_path.append(current_node.coordinates)
            current_node = cls.map_nodes[current_node.parent]
        cls.best_path.append(current_node.coordinates)
        print cls.best_path
        plt.show()
        return cls.best_path[::-1]
    
    @classmethod
    def nearest_node(cls, coords):
        '''
        Finds the index, distance and angle of the nearest node to some input coordinates.
        
        Input:
        coords: desired coordinates as (xval, yval) tuple

        Returns:
        (NODE# ,(DISTANCE, ANGLE))

        Angle is provided in radians with 0 on the positive x axis
        Node number is the index in the library of nodes
        '''
        dists = []
        for node in cls.map_nodes.values():
            dists.append((node.node_index, node.distance(coords)))
        nearest = min(dists, key = lambda x : x[1][0])
        # print 'Nearest Node Selected: {}'.format(nearest)
        return nearest

    @classmethod
    def new_node(cls, nearest, stp_sz, animate=True):
        '''
        Establishes a new node at a distance of a given step size
        
        Input:
        nearest : (Node_num, (distance, radian_angle))
            Returned format and values from the nearest_node classmethod
        stp_sz : Distance to attempt to establish a new node from
        '''
        # Establish node object as basis for new node generation
        anchor = cls.map_nodes[nearest[0]]
        # Creates new x,y location from the values generated from the nearest node function
        x_new = anchor.xloc + stp_sz*math.cos(nearest[1][1])
        y_new = anchor.yloc + stp_sz*math.sin(nearest[1][1])
        # Check for validity of a new point
        if anchor.valid_check(x_new, y_new):
            # Create a new node at the valid location
            cls(x_new, y_new, parent=nearest[0])
            # Draw the new animated circle and pause
            if animate:
                cls.draw_circ((x_new, y_new), cir_color='#CCCCCC')
                plt.pause(0.05)
        



PATHFIND.set_map(map_size, 1.0)
PATHFIND.load_obstacles(obstacles, 1.0)
PATHFIND.set_goal(goal_pos)
PATHFIND.set_start(start_pos)
# PATHFIND.run_RRT(1.0, animate=True)
def get_rrt_path(step, ani=True):
    nodes = PATHFIND.run_RRT(step, animate=ani)

    PATHFIND.draw_map('Rapidly Exploring Random Trees (RRT)')
    linex = [i[0] for i in nodes[1:]]
    liney = [i[1] for i in nodes[1:]]
    plt.plot(linex, liney, color='b', linewidth=2)
    figName = get_filename('proj', '.png', './log_files/')
    plt.savefig(figName, bbox_inches='tight')
    plt.show()
    return nodes

def optim1(step, ani=True):
    nodes = PATHFIND.optimize1(step, animate=ani)
    PATHFIND.draw_map('Optimize Attempt 1')
    linex = [i[0] for i in nodes[1:]]
    liney = [i[1] for i in nodes[1:]]
    plt.plot(linex, liney, color='b', linewidth=2)
    figName = get_filename('proj', '.png', './log_files/')
    plt.savefig(figName, bbox_inches='tight')
    plt.show()
    return nodes
    
get_rrt_path(1.0)
# optim1(1.0)
