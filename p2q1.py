#G6T18
#Tan Chin Hoong

# project 2 Q1

# replace the content of this function with your own algorithm
# inputs: 
#   p: min target no. of points player must collect. p>0
#   v: 1 (non-cycle) or 2 (cycle)
#   flags: 2D list [[flagID, value, x, y], [flagID, value, x, y]....]
# returns:
#   1D list of flagIDs to represent a route. e.g. [F002, F005, F003, F009]
from sklearn.metrics.pairwise import euclidean_distances
import copy
import random
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def get_route(p, v, flags):
  # code here
	routesArray = []
	holder = {}
	pointsGotten = 0
	flagsDict = generate_flags_dict(flags)
	extra = copy.deepcopy(flagsDict)
	currentPoint = ["F000",0,0,0]
	while pointsGotten < p:
		allFlags = []
		for i in extra:
			allFlags.append(int(extra[i][1])/float(get_distance(currentPoint,extra[i])))
		indexOfMax = allFlags.index(max(allFlags))
		keyOfIndex = list(extra)[indexOfMax]
		pointsGotten += flagsDict[keyOfIndex][1]
		print(pointsGotten)
		currentPoint = extra[keyOfIndex]
		allFlags.pop(indexOfMax)
		del extra[keyOfIndex]
		routesArray.append(keyOfIndex)


	best = routesArray
	print(routesArray)
	bestD = get_dist_and_points_q1(best,flagsDict,v)
	for i in range(0, len(best)):
		for j in range(i+1, len(best)):
			updatedRoute = swapSides(i,j,best)
			newDistance = get_dist_and_points_q1(updatedRoute,flagsDict,v)

			if (newDistance < bestD):
				best = updatedRoute
				bestD = newDistance
	currentRoute = best
	bestRoute = removeDiff(best,currentRoute,bestD,pointsGotten-p,flagsDict,v)
	return bestRoute


def swapSides(front,end, routesArray):
	returnArray = routesArray[:front] + routesArray[front:end+1][::-1] + routesArray[end+1:]
	return returnArray


def twoOpt(route,v,flagsDict):
	best = route
	bestD = get_dist_and_points_q1(best,flagsDict,v)
	for i in range(0, len(route)):
		for j in range(i+1, len(route)):
			updatedRoute = swapSides(i,j,best)
			newDistance = get_dist_and_points_q1(updatedRoute,flagsDict,v)

			if (newDistance < bestD):
				best = updatedRoute
				bestD = newDistance
	
	return best
def removeDiff(startingRoute, bestRoute, bestdistance, difference, flagsdict, v):
    if difference > 0:
        # Get a list of sorted array with all the route that have point <= the required points
        sorted_route = sorted([i for i in flagsdict if flagsdict[i][1] <= difference], reverse=True)
        for remove_extra in sorted_route:
            # remove the flags with the higest point
            updatedRoute = [point for point in startingRoute if point != remove_extra]
            newDistance = get_dist_and_points_q1(updatedRoute, flagsdict, v)
            if (newDistance < bestdistance):
                bestRoute = updatedRoute
                bestdistance = newDistance
    return bestRoute   
def get_dist_and_points_q1(your_route, flags_dict, v, verbose=False):

  # check for syntax error first


  # calculate distance and points
  dist = 0
  points = 0

  start_node = ["Start", 0, 0, 0] # starting point (0, 0)
  last_node = start_node
  
  for flagID in your_route:
    if not flagID in flags_dict:
      return "Flag ID in your route is not valid : " + flagID, 0, 0 # error

    curr_node = flags_dict[flagID]
    dist_to_curr_node = get_distance(last_node, curr_node)
    dist += dist_to_curr_node
    points += curr_node[1]
    
    if verbose:
      print("last_node:" + str(last_node) + ", curr_node:" + str(curr_node))
      print("dist_to_curr_node:" + str(dist_to_curr_node))
      print("dist so far:" + str(dist) +", points so far:" + str(points) + "\n---")

    last_node = curr_node
  # to go back to SP?
  if v == 2:   # cycle back to SP
    dist += get_distance(last_node, start_node)
    if verbose: print("v = 2, so go back to SP")

  if verbose:
    print("final dist for this route:" + str(dist) + "\n---")
    
  return None, dist, points # no error

def get_distance(node_A, node_B):
  return ((float(node_A[2]) - float(node_B[2])) ** 2 + (float(node_A[3]) - float(node_B[3])) ** 2) ** 0.5
def generate_flags_dict(flags_list):
  d = {}
  for item in flags_list:
    #             flagID,  points,       x,              y
    d[item[0]] = [item[0], int(item[1]), float(item[2]), float(item[3])]
  return d