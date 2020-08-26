# <Your Team ID>G6T18
# <Team members' names>Tan Chin Hoong

# project 2 Q2

# replace the content of this function with your own algorithm
# inputs: 
#   p: min target no. of points team must collect. p>0
#   v: 1 (non-cycle) or 2 (cycle)
#   flags: 2D list [[flagID, value, x, y], [flagID, value, x, y]....]
# returns:
#   A list of n lists. Each "inner list" represents a route. There must be n routes in your answer
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import timeit
import copy
def get_routes(p, v, flags, n):
  # code here
  # return [["F001", "F002", "F003"], ["F009", "F006"]]
    flags_dict = {}
    if(n == 1):
        return [get_route(p,v,flags)]
    if p<=120:
        return [get_route(p,v,flags)] + [[] for i in range(n-1)]
    
    for i in flags:
        flags_dict[i[0]] = i[2:]
    flagsDict = generate_flags_dict(flags)
    # start = timeit.default_timer()
    x = get_k_splits(flags_dict,n)
    # stop = timeit.default_timer()
    # print(stop-start)
    count = 0
    clusterDict = {}
    flags.insert(0, ['F000', '0', '0.00000000', '0.00000000'])
    for a in x:
        ratioDict = {}
        inter = [0] + a
        for i in inter :
            current_flag = flags[i]
            ratioDict[convertStringToInt(current_flag[0])] = []
            middle = []
            for j in inter:
                if i != j:
                    other_flag = flags[j]
                    distance = get_distance(current_flag, other_flag)
                    weighted_value = float(other_flag[1])/distance
                    ratioDict[convertStringToInt(current_flag[0])].append(weighted_value) # append the name and weighted value
                else:
                    ratioDict[convertStringToInt(current_flag[0])].append(0)
            clusterDict[count] = ratioDict
        count+=1
    pointsGotten = 0
    holder = [[0] for x in range(n)]
    overallList = []
    while pointsGotten < p:
        mostValue = 0
        bestCandidateCluster = 0
        candidateFlag = 0
        indexOfBestCandidate = 0
        for clusters in range(len(x)):
            current = holder[clusters]
            lastGuy = current[-1]
            toInspect = clusterDict[clusters]
            wantToCheck = toInspect[lastGuy]
            highestRatio = max(wantToCheck)
            if(highestRatio > mostValue):
                mostValue = highestRatio
                indexOfBestCandidate = wantToCheck.index(highestRatio)
                candidateFlag = x[clusters][indexOfBestCandidate-1]
                bestCandidateCluster = clusters
        for i in clusterDict[bestCandidateCluster]:
            dictionaryToReset = clusterDict[bestCandidateCluster]
            for j in dictionaryToReset:
                dictionaryToReset[j][indexOfBestCandidate] = 0
        holder[bestCandidateCluster].append(candidateFlag)
        overallList.append(candidateFlag)
        points = int(flags[candidateFlag][1])
        pointsGotten+= int(flags[candidateFlag][1])
        
    distances = []
    for i in flags:
        middle = []
        for j in flags:
            if j[0]==i[0]:
                middle.append(0)
            else:
                middle.append(get_distance(i,j))
        distances.append(middle)
    
    toReturn = []
    anotherHolder = []
    final = twoOptQ2(holder, distances,v,p)

    for i in final:
        extra=[]
        for x in i[1:]:
            extra.append(flags[x][0])
        toReturn.append(extra)
    return toReturn
    # print(toReturn)
            

# def twoOpt(mid,end, routesArray):
#     toReturn = []
#     for i in range(0, mid):
#         toReturn.append(routesArray[i])
#     counter = 0
#     for i in range(end, mid-1, -1):
#         toReturn.append(routesArray[i])
#     for i in range(end+1,len(routesArray)):
#         toReturn.append(routesArray[i])
        
#     return toReturn

# def twoOpt( mid,end, routesArray):
#     returnArray = routesArray[:]
#     returnArray[:mid] = routesArray[:mid]
#     returnArray[mid:end+1] = routesArray[end:mid-1:-1]
#     returnArray[end+1:] = routesArray[end+1:]        
#     return returnArray

def swapSides( mid,end, routesArray):
    returnArray = routesArray[:]
    returnArray[:mid] = routesArray[:mid]
    returnArray[mid:end+1] = routesArray[end:mid-1:-1]
    returnArray[end+1:] = routesArray[end+1:]        
    return returnArray

def twoOptQ2(arrayOfRoutes,distances,v,p):
    # print(p)
    finalReturnArray = []
    # print(arrayOfRoutes)
    for index, route in enumerate(arrayOfRoutes): 
        route = route[1:] 
        startLen = len(route)
        if(len(route) >1):
            best = route[:]
            # print(best)
            indexes = list(range(len(route)))
            improved = True
            while improved:
                # print(best)
                improved = False
                for y in list(range(5,0,-1)):
                    for start in indexes[1:]:
                        end = (start + y) % len(route)
                        route = best[:]
                        if(start<end):
                            sectionFlipped = route[start:end+1][::-1]
                            route[start:end+1] = sectionFlipped
                        else:
                            route = swapSides(start,end+1, route)
                        # best = route [:]

                        endDist = distances[0][route[-1]]
                        endDistBest = distances[0][best[-1]]
                        if v == 2:
                            if(calcDist(distances, route) + endDist + distances[0][route[0]] < calcDist(distances,best)+endDistBest + distances[0][route[0]]):
                            # if(v == 2):
                            #     route.pop(len(route)-1)
                            #     best.pop(len(best)-1)
                                best = route
                                improved = True
                        else:
                            if(calcDist(distances, route) + distances[0][route[0]] < calcDist(distances,best) + distances[0][route[0]]):
                            # if(v == 2):
                            #     route.pop(len(route)-1)
                            #     best.pop(len(best)-1)
                                best = route
                                improved = True
            finalReturnArray.append([0]+best)

        else:
            finalReturnArray.append(route)
    return finalReturnArray

def calcDist(distances, routes):
    distance = 0
    # print(routes[0], routes[1])
    for i in range(1,len(routes)):
        # print(distances[i-1])
        distance += distances[routes[i-1]][routes[i]]
    return distance

def convertStringToInt(x):
    toReturn = ""
    for i in x:
        if i == 'F':
            pass
        else:
            toReturn = toReturn + i
    return int(toReturn)
def get_k_splits(flags_dict, n):
    flags = pd.DataFrame.from_dict(flags_dict, orient="index")
    flags.columns= ['x', 'y']
    # print(orders)
   

    x = dict()
    count = 0
    best = n
    sse = {}
    for i in range(1,6):
        model5 = KMeans(n_clusters = i)
        model5.fit(flags[['x','y']])
        sse[i] = model5.inertia_
    for i in range(1,5):
        percentage = (sse[i] - sse[i+1])/(sse[i])
        if(percentage > count):
            count = percentage
            best = i+1
    # print(best)
    if(n<best):
        best = n
    final_clust = KMeans(n_clusters=best, random_state=10, max_iter=200).fit(flags[['x','y']])
    def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]
    for cluster_no in range(best):
        x[cluster_no] = list(ClusterIndicesNumpy(cluster_no, final_clust.labels_))
    groups = [z for z in x.values()]
    paths = []
    for order_arr in groups:
        order_ids = []
        for order_id in order_arr:
            order_ids.append(order_id + 1)
        paths.append(order_ids)
    return paths

def get_distance(node_A, node_B):
  return  ( (float(node_A[2]) - float(node_B[2]))** 2 + ((float(node_A[3]) - float(node_B[3])) ** 2) ) ** 0.5
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
		# print(pointsGotten)
		currentPoint = extra[keyOfIndex]
		allFlags.pop(indexOfMax)
		del extra[keyOfIndex]
		routesArray.append(keyOfIndex)


	best = routesArray
	# print(routesArray)
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