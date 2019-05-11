#!/usr/bin/env python

## module rrt_search
import os, csv, math, datetime, random
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
map_size = [(0.0,0.0),(15.0,15.0)]

start_pos = (1.0,1.0)

goal_pos = (7.0,13.0)

obstacles = [(2.0,3.0),
             (2.0,4.0),
             (2.0,5.0),
             (0.0,5.0),
             (1.0,5.0),
             (2.0,5.0),
             (3.0,5.0),
             (4.0,5.0),
             (5.0,5.0),
             (5.0,4.0),
             (5.0,3.0),
             (8.0,2.0),
             (9.0,2.0),
             (10.0,2.0),
             (11.0,2.0),
             (12.0,2.0),
             (13.0,2.0),
             (8.0,3.0),
             (8.0,4.0),
             (8.0,5.0),
             (8.0,6.0),
             (8.0,7.0),
             (8.0,8.0),
             (8.0,9.0),
             (8.0,7.0),
             (2.0,7.0),
             (3.0,7.0),
             (4.0,7.0),
             (5.0,7.0),
             (6.0,7.0),
             (7.0,7.0),
             (9.0,6.0),
             (10.0,6.0),
             (11.0,6.0),
             (12.0,6.0),
             (13.0,6.0),
             (14.0,6.0),
             (15.0,6.0),
             (2.0,8.0),
             (2.0,9.0),
             (2.0,10.0),
             (2.0,11.0),
             (2.0,12.0),
             (2.0,13.0),
             (5.0,9.0),
             (5.0,10.0),
             (5.0,11.0),
             (5.0,12.0),
             (5.0,13.0),
             (5.0,14.0),
             (5.0,15.0),
             (6.0,12.0),
             (7.0,12.0),
             (8.0,12.0),
             (9.0,12.0),
             (10.0,12.0),
             (11.0,12.0),
             (12.0,8.0),
             (12.0,9.0),
             (12.0,10.0),
             (12.0,11.0),
             (12.0,12.0)]

dist = lambda a, b, c, d : ((abs(c - a)**2 + abs(d - b)**2)**0.5)

class RRT_Node:
    '''
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

    RRT_Node.set_map(map_size, block_size)
    RRT_Node.load_obstacles(obstacles, obs_size)
    RRT_Node.set_goal(goal_pos)
    RRT_Node.set_start(start_pos)
    RRT_Node.run_rrt(stp_size)
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
            RRT_Node.num_nodes += 1
            self.xloc = x_coord
            self.yloc = y_coord
            self.coordinates = (x_coord, y_coord)
            self.cost = 0.0
            self.parent = parent
            self.start_node = start
            self.node_index = RRT_Node.num_nodes
            RRT_Node.map_nodes[RRT_Node.num_nodes] = self
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
        if RRT_Node.x_min <= x_val <= RRT_Node.x_max and RRT_Node.y_min <= y_val <= RRT_Node.y_max:
            for obs in RRT_Node.map_obs:
                if obs[0][0] <= x_val <= obs[1][0] and obs[0][1] <= y_val <= obs[1][1]:
                    print 'valid_check : ({:1.2f},{:1.2f}) collision with obstacle at [({:1.2f},{:1.2f}),({:1.2f},{:1.2f})]'.format(x_val, y_val, obs[0][0], obs[0][1], obs[1][0], obs[1][1])
                    # in_cmd = raw_input('valid_check : ({:1.2f},{:1.2f}) collision with obstacle at [({:1.2f},{:1.2f}),({:1.2f},{:1.2f})]'.format(x_val, y_val, obs[0][0], obs[0][1], obs[1][0], obs[1][1]))
                    # if in_cmd == 's':
                    #     exit()
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
        cls.end_goal = coords

    @classmethod
    def set_start(cls, coords):
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
            plt.scatter(node.xloc, node.yloc, color='#CCCCCC')

        for key, point in enumerate(cls.best_path[:-1]):
            lin_x = [point[0], cls.best_path[key+1][0]]
            lin_y = [point[1], cls.best_path[key+1][1]]
            plt.plot(lin_x, lin_y, color='#4682B4', linewidth=2)
        cls.ax.set_ylabel('y location', fontsize=12)
        cls.ax.set_xlabel('x location', fontsize=12)
        cls.ax.set_title(title + ' {} Nodes Generated.'.format(cls.num_nodes))
        plt.axis('equal')
        plt.axis([cls.x_min - 1.0,cls.x_max + 1.0,cls.y_min - 1.0,cls.y_max + 1.0])
        plt.tight_layout()

    @classmethod
    def draw_border(cls):
        box_color = '#555555'
        ln_width = 3
        cls.ax.plot([cls.x_min, cls.x_max],[cls.y_min, cls.y_min], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_min, cls.x_max],[cls.y_max, cls.y_max], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_min, cls.x_min],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)
        cls.ax.plot([cls.x_max, cls.x_max],[cls.y_min, cls.y_max], color=box_color, linewidth=ln_width)

    @classmethod
    def draw_obstacles(cls):
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
        circ = [mpatches.Circle(coords, radius=rad)]
        pc = PatchCollection(circ, facecolor=cir_color, edgecolor='None')
        cls.ax.add_collection(pc)

    @classmethod
    def run_RRT(cls, step_size, animate=False):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~  RUN_RRT CALLED~~~~~~~~~~~~~~~~~~~~~~'
        # Set plot space header while we are trying to find this path
        if animate: cls.draw_map('Processing RRT...')
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
    def nearest_node(cls, coords):
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
        # x_new = round(anchor.xloc + stp_sz*math.cos(nearest[1][1]))
        # y_new = round(anchor.yloc + stp_sz*math.sin(nearest[1][1]))
        # Check for validity of a new point
        if anchor.valid_check(x_new, y_new):
            # Create a new node at the valid location
            cls(x_new, y_new, parent=nearest[0])
            # Draw the new animated circle and pause
            if animate:
                cls.draw_circ((x_new, y_new), cir_color='#808080')
                plt.pause(0.05)
        

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

RRT_Node.set_map(map_size, 1.0)
RRT_Node.load_obstacles(obstacles, 1.0)
RRT_Node.set_goal(goal_pos)
RRT_Node.set_start(start_pos)
# RRT_Node.run_RRT(1.0, animate=True)

def get_path(step, ani=False):
    nodes = RRT_Node.run_RRT(step, animate=ani)

    RRT_Node.draw_map("Maze RRT Comparison (optimized)")

    figName = get_filename('exam_rrt_optimized', '.png', './log_files/')
    plt.savefig(figName, bbox_inches='tight')
    plt.show()
    return nodes

get_path(1.0, True)