'''
graph = (Vertices, Edges)

Adjacency List (check if nodes are connected: O(n))
    start edge: connected to
    1: [2,5]
    2: [3]
    3: [2,4,5,6]

    Weighted adjacency list
    1: [(2, weight 1), (5, weight 2)]

Adjacency Hash Set - same as list but O(1) time

Adjacency Matrix - search: O(n^2)

Depth First Search  O(vertices + edges)
    when looking at a node, pick its first link and investigate that node's first link, etc. If target not found, then return back and choose the next unexplored option

    Store array of bool's called visited

    DFS Starter() #Make sure every node is visited even if disconnected from the rest of the graph
    for all vertices in graph: if not visited, DFS(graph,vertex)

    DFS(Graph, v)
    visited[v] = true
    for neighbor in v:
        if not visited: DFS(Graph, neighbor)

        

    
u v y x
'''

def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
    deg = [0 for _ in range(n)]
    memo = set()
    for a,b in roads:
        #traverse every edge and update degree
        deg[a] += 1
        deg[b] += 1
        memo.add((min(a,b),max(a,b))) #add sorted edge to memo of shared connections
    
    ans = 0
    #for every pair, evaluate value as sum of degrees minus shared connections
    for i in range(n):
        for j in range(i+1, n):
            ans = max(ans, deg[i] + deg[j] - ((i,j) in memo))
    return ans

def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
    #Multiply the node's number by the number of connecting roads, which represents how many times that number contributes to the total
    deg = [0 for _ in range(n)]
    for a, b in roads:
        deg[a] += 1
        deg[b] += 1
    
    deg.sort()
    return sum([(i+1)*deg[i] for i in range(n)])

#Breadth First Search O(#vertices + #edges)
'''
#start with queue containing only the starting node s
q = [s] #Q is the list of nodes to search next
visited = []
#d is an array with index = nodeNumber and value=depth. it is the program output.
d = [mathf.inf for _ in nodes] #??
D=0 #depth
while q:
    size = len(q)
    while size:
        size -= 1
        node = q.pop(0)
        d[node] = D
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
                #Set predecessor of this node to the previous node
    D += 1    
'''


#Function to return the number of nodes reachable in a graph with "restricted" off-limits nodes
def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
    visited = set(restricted) #Hash set that only includes visited nodes. Restricted nodes have been 'already visited' and should not be explored

    visited.add(0)
    adj = dict()
    for a, b in edges:
        if a not in adj:
            adj[a] = list()
        if b not in adj:
            adj[b] = list()
        adj[a].append(b)
        adj[b].append(a)

    ans = 0
    def depthFirstSearch(node):
        nonlocal ans
        ans += 1
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                depthFirstSearch(neighbor)
        return

    depthFirstSearch(0)
    return ans
            
#Code with a graph of courses and prerequisites, returns true if a student can finish the courses
#Kahn's algorighm
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    inDegrees = [0 for _ in range(numCourses)]
    adj = {i: list() for i in range(numCourses)}
    for a, b in prerequisites:
        inDegrees[a] += 1
        adj[b].append(a)
    
    q = [i for i in range(numCourses) if inDegrees[i] == 0] #Queue of all courses with no prerequisites
    counter = 0
    while q:
        node = q.pop(0)
        counter += 1
        #go to the destination nodes and decrease their num of prereq's by one (you completed the course)
        for neighbor in adj[node]:
            inDegrees[neighbor] -= 1
            if inDegrees[neighbor] == 0:
                q.append(neighbor) #add the courses that now have no prereq's
    return counter == numCourses

#In a grid of "0" and "1", count the number of Islands (connected groups of 1's)
def numIslands(self, grid: List[List[str]]) -> int:
        directions = [-1,0,1,0,-1] #First check -1,0. Then 0,1 then 1,0 then 0,-1
        #directionsAllowDiagonal = [-1,0,1,0,-1,-1,1,1,-1]
        m, n = len(grid), len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]
        ans = 0

        for i in range(m):
            for j in range(n):
                if(grid[i][j] == "1" and not visited[i][j]):
                    ans += 1

                    #Breadth first search
                    q = [(i,j)]
                    visited[i][j] = True
                    while q:
                        x, y = q.pop(0)
                        for k in range(4): #Use consecutive pairs in Directions to check above, below, sides
                            neighborX, neighborY = x+directions[k], y+directions[k+1]
                            if 0 <= neighborX < m and 0 <= neighborY < n and grid[neighborX][neighborY] == "1" and visited[neighborX][neighborY] == False:
                                visited[neighborX][neighborY] = True
                                q.append((neighborX,neighborY))
        return ans

#Return the least number of Perfect Squares needed to add to number n. Build a graph of all numbers, whose connections are between perfect squares
#Breadth-first search used
def numSquares(self, n: int) -> int:
    memo = list()
    i=1
    while i*i <=n: #generate perfect squares
        memo.append(i*i)
        i += 1
    visited = set(memo)
    
    q = memo.copy() #start with a graph of perfect squares with depth/dist=1
    dist = 1
    while q:
        next_q = list()
        for v in q:
            if v == n: #max distance
                return dist
            else:
                for w in memo:
                    temp = v+w
                    if temp <= n and temp not in visited: #only if reachable
                        visited.add(temp)
                        next_q.append(temp)
        q = next_q
        dist += 1 #go one layer deeper into the search

#Program given an array of coin values, to figure out the least amount of coins needed to make a value n
#Using b-f-s shortest-path on a grid connected by coin values
def coinChange(self, coins: List[int], amount: int) -> int:
    q = [0]
    visited = {0}
    dist = 0

    while q:
        size = len(q)
        while size:
            size -= 1
            v = q.pop(0)
            if v == amount: #you've reached target
                return dist
            for coin in coins:
                temp = v + coin
                if temp <= amount and temp not in visited: #if temp is within the grid size
                    visited.add(temp)
                    q.append(temp)
        dist += 1
    #in case there is no path
    return -1

#Flood Fill (doesn't work completely)
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    directions = [-1,0,1,0,-1] #First check -1,0. Then 0,1 then 1,0 then 0,-1
    m, n = len(image), len(image[0])
    visited = [[False for _ in range(n)] for _ in range(m)]

    #Breadth first search
    q = list()
    q.append((sr,sc))
    startColor = image[sr][sc]
    visited[sr][sc] = True
    while q:
        x, y = q.pop(0)
        print(f"({x},{y})")
        image[x][y] = color
        for k in range(4): #Use consecutive pairs in Directions to check above, below, sides
            neighborX, neighborY = x+directions[k], y+directions[k+1]
            if 0 <= neighborX < m and 0 <= neighborY < n and image[neighborX][neighborY] == startColor and visited[neighborX][neighborY] == False:
                visited[neighborX][neighborY] = True
                q.append((neighborX,neighborY))
    return image