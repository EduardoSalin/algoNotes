'''
5. Longest Palindromic Substring

Intuition
To find the longest palindromic substring, you can use the fact that palindromes mirror around their center. For each character in the string, consider it as the center of a potential palindrome. Expand outwards to check the longest palindrome centered at that character. This needs to be done for both odd-length and even-length palindromes.

Approach
Expand Around Center: For each character in the string at index i, consider two cases for expansion:
An odd-length palindrome centered at i (e.g., abcba with c at the center).
An even-length palindrome with its center as indices i and i+1 (e.g., abba with the center between the two bs).
Expand outward from these centers as long as the characters on both sides are equal. This process will help determine the length of the longest palindrome for each center.
Track Maximum Length: Keep track of the maximum length palindrome and its start and end indices while expanding around each possible center.
'''
class Solution:
    # @param A : string
    # @return a strings
    def findLps(self, A, i,j, N):
        count = 0
        if i < 0 and j >= N:
            return (count, i, j)
        while i >= 0 and j < N :
            if A[i] == A[j]:
                if i == j:
                    count += 1
                else:
                    count += 2
                i -= 1 
                j += 1
            else:
                break
        return (count, i+1, j-1)


    def longestPalindrome(self, A: str) -> str:
        N = len(A)
        max_len = 1
        s = 0
        e = 0
        for i in range(len(A)):
            # odd length pallindrome
            length, l, r = self.findLps(A,i,i, N)
            if length > max_len:
                max_len = length
                s = l 
                e = r

            # even length pallindrome
            length, l, r = self.findLps(A,i,i+1, N)
            if length > max_len:
                max_len = length
                s = l 
                e = r

        return A[s:e+1]

        
'''
45. Jump Game II
Intuition :
We have to find the minimum number of jumps required to reach the end of a given array of non-negative integers i.e the shortest number of jumps needed to reach the end of an array of numbers.
Explanation to Approach :
We are using a search algorithm that works by moving forward in steps and counting each step as a jump.
The algorithm keeps track of the farthest reachable position at each step and updates the number of jumps needed to reach that farthest position.
The algorithm returns the minimum number of jumps needed to reach the end of the array.
'''
class Solution:
  def jump(self, nums: List[int]) -> int:
    ans = 0
    end = 0
    farthest = 0

    # Implicit BFS
    for i in range(len(nums) - 1):
      farthest = max(farthest, i + nums[i])
      if farthest >= len(nums) - 1:
        ans += 1
        break
      if i == end:      # Visited all the items on the current level
        ans += 1        # Increment the level
        end = farthest  # Make the queue size for the next level

    return ans
  
'''
62. Unique Paths
Intuition
The main idea to solve the unique paths question is to use dynamic programming to calculate the number of unique paths from the top-left corner of a grid to the bottom-right corner.

This approach efficiently solves the problem by breaking it down into smaller subproblems and avoiding redundant calculations. It's a classic example of dynamic programming used for finding solutions to grid-related problems.
'''
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        aboveRow = [1] * n

        for _ in range(m - 1):
            currentRow = [1] * n
            for i in range(1, n):
                currentRow[i] = currentRow[i-1] + aboveRow[i]
            aboveRow = currentRow
        
        return aboveRow[-1]

'''
63. Unique Paths II
'''

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:              
        m, n = len(obstacleGrid), len(obstacleGrid[0])        
        
        dp=[[0] * (n+1) for _ in range(m+1)]        
        dp[0][1]=1
                        
        for row in range(1, m+1):
            for col in range(1, n+1):
                if not obstacleGrid[row-1][col-1]:
                    dp[row][col] = dp[row-1][col] + dp[row][col-1]
         
        return dp[-1][-1]
    
'''
64. Minimum Path Sum
Approach:
The code implements a dynamic programming approach to find the minimum path sum in a grid.

The algorithm uses a 2D array to store the minimum path sum to reach each position (i, j) in the grid, where i represents the row and j represents the column.

The minimum path sum to reach each position (i, j) is computed by taking the minimum of the path sum to reach the position above (i-1, j) and the position to the left (i, j-1), and adding the cost of the current position (i, j).

The minimum path sum to reach the bottom-right corner of the grid is stored in the last element of the array (grid[m-1][n-1]), where m is the number of rows and n is the number of columns in the grid.

Intuition:
The intuition behind the dynamic programming approach is that the minimum path sum to reach a position (i, j) in the grid can be computed by considering the minimum path sum to reach the positions (i-1, j) and (i, j-1).

This is because the only two possible ways to reach the position (i, j) are either by moving down from (i-1, j) or moving right from (i, j-1).

By computing the minimum path sum to reach each position in the grid, the algorithm can find the minimum path sum to reach the bottom-right corner of the grid by simply looking at the last element of the array (grid[m-1][n-1]).
'''
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
            
        
        m, n = len(grid), len(grid[0])
        
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[-1][-1]
    
'''
72. Edit Distance
Intuition :
Here we have to find the minimum edit distance problem between two strings word1 and word2.
The minimum edit distance is defined as the minimum number of operations required to transform one string into another.
Approach :
The approach here that I am using is dynamic programming. The idea is to build a 2D matrix dp where dp[i][j] represents the minimum number of operations required to transform the substring word1[0...i-1] into the substring word2[0...j-1].

'''
class Solution:
  def minDistance(self, word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    # dp[i][j] := min # Of operations to convert word1[0..i) to word2[0..j)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
      dp[i][0] = i

    for j in range(1, n + 1):
      dp[0][j] = j

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if word1[i - 1] == word2[j - 1]:
          dp[i][j] = dp[i - 1][j - 1]
        else:
          dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    return dp[m][n]
  

'''
198. House Robber

Goal:
Find the maximum amount that can be robbed from a row of houses, with the constraint of not robbing adjacent houses.

How we Achieved it:
By utilizing dynamic programming, we maintain two variables (s0 and s1) to represent the maximum amount when the current house is not robbed and when it is robbed. Iterating through the houses and updating these variables at each step, we dynamically calculate the optimal solution, ensuring the maximum amount considering the constraints. The final result is the maximum value between s0 and s1.

Approaches
(Also explained in the code)

Utilizes dynamic programming to determine the maximum amount that can be robbed from a row of houses.
Uses two vectors (s0 and s1) to track the maximum amount without and with robbing the current house, respectively.
Iterates through the houses, updating s0 and s1 at each step based on the decision to rob or not rob the current house.
Returns the maximum value be
'''

class Solution:
    def rob(self, nums):
        n = len(nums)
        if n == 0:
            return 0

        s0, s1 = [0, 0], [0, 0]
        s1[0] = nums[0]

        for i in range(1, n):
            s0[i % 2] = max(s0[(i - 1) % 2], s1[(i - 1) % 2])
            s1[i % 2] = s0[(i - 1) % 2] + nums[i]

        return max(s0[(n - 1) % 2], s1[(n - 1) % 2])

'''
213. House Robber II

Intuition
In recursive Apporoach u will get a TLE in this question so DP is an optimized acceptable solution her.
Base Cases
1. If only 1 house present return it as max amount to be robbed
2. If only 2 houses present return the max of them

How this question is different from House Robbers I?
Because the houses are in circular manner so u can go -
From 0 => n - 2 (last House excluded as in circular it will be adjacent to the first house and we cant rob it)
From 1 => n - 1
So for each choice define a dp and find the max item u can rob exact similarly like house robbers I and from both choices i.e starting from either 0 index or 1 index choose which gives the max ans.

'''

class Solution:
    def rob(self, arr: List[int]) -> int:
        n = len(arr)
        if n == 1:
            return arr[0]
        if n == 2:
            return max(arr[0], arr[1])
        def maxRob(lastRange, startRange):
            dp = [0] * (n + 1) # for each choice a unique dp defined
            for i in range(startRange, lastRange):
                if dp[i] != 0:
                    return dp[i]
                pick = arr[i] + dp[i-2] #pick it & then check for next to next
                notPick = 0 + dp[i-1] #dont pick check for next
                dp[i] = max(notPick, pick)
            # print(max(dp)) -->
            return max(dp)
        range1 = maxRob(n-1, 0)
        range2 = maxRob(n, 1)
        # print(range1, range2)
        return max(range1, range2)
    

'''
337. House Robber III

we construct a dp tree, each node in dp tree represents [rob the current node how much you gain, skip the current node how much you gain]
dp_node[0] =[rob the current node how much you gain]
dp_node[1] =[skip the current node how much you gain]
we start the stolen from the leaf: Depth First Search
for each node you have 2 opitions:
option 1: rob the node, then you can't rob the child of the node.
dp_node[0] = node.val + dp_node.left[1] +dp_node.right[1]
option 2: skip the node, then you can rob or skip the child of the node.
dp_node[1] = dp_node.left[0] + dp_node.right[0]
the maximum of gain of the node depents on max(dp_node[0],dp_node[1])
'''
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        return max(self.dfs(root))
    
    def dfs(self, root: TreeNode):
        if not root:
            return (0, 0)
        left = self.dfs(root.left)
        right = self.dfs(root.right)
        return (root.val + left[1] + right[1], max(left[0], left[1]) + max(right[0], right[1]))

'''
413. Arithmetic Slices
Intuition
Here we have list of nums.
Our goal is to find count of distinct ways to extract arithmetic subarrays.

Approach
A subarray is valid, if it follows to i < i + 1 < i + 2 ... < n for each i index and diff == nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2].
This will be the recurrence relation for DP.
'''
class Solution:
    def numberOfArithmeticSlices(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [0] * n

        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1

        return sum(dp)
    

'''
823. Binary Trees With Factors
Intuition :
In order to build a tree, we need to know the number of left children and right children.
If we know the number of left children and right children, the total number of trees we can build is left * right.
So we can use a hashmap to store the number of trees we can build for each number.
We iterate the array from small to big, for each number, we iterate all the numbers smaller than it to find the left children,
and we check if the right children exists in the hashmap.
If so, we can build left * right trees.
Finally, we sum up the number of trees for each number.

Algorithm :
Sort the array "arr", and use a hashmap dp to record the number of trees we can build for each number.
Iterate the array from small to big, for each number, we iterate all the numbers smaller than it to find the left children, and we check if the right children exists in the hashmap. If so, we can build left * right trees.
Finally, we sum up the number of trees for each number.
'''
class Solution:
    def numFactoredBinaryTrees(self, arr: List[int]) -> int:
        arr.sort()
        dp = {}
        for i in range(len(arr)):
            dp[arr[i]] = 1
            for j in range(i):
                if arr[i] % arr[j] == 0 and arr[i] // arr[j] in dp:
                    dp[arr[i]] += dp[arr[j]] * dp[arr[i] // arr[j]]
        return sum(dp.values()) % (10**9 + 7)
    
'''
103. Binary Tree Zigzag Level Order Traversal


'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        b=[root]
        res=[]
        flag=True
        while b:
            val=[]
            b1=[]
            for i in b:
                val.append(i.val)
                if i.left:
                    b1.append(i.left)
                if i.right:
                    b1.append(i.right)
            if flag:
                res.append(val)
                flag=False
            else:
                res.append(val[::-1])
                flag=True
            b=b1
        return res

'''
200. Number of Islands
bfs
'''
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:return 0
        rows,cols=len(grid),len(grid[0])
        count,visit=0,set()
        def bfs(r,c):
            q=deque()
            q.append((r,c))
            visit.add((r,c))
            while q:
                row,col=q.popleft()
                direction=[[1,0],[-1,0],[0,1],[0,-1]]
                for dr,dc in direction:
                     r,c=row+dr,col+dc
                     if (r in range(rows) and c in range(cols) and (r,c) not in visit and grid[r][c]=="1"):
                         q.append((r,c))
                         visit.add((r,c))
                     
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]=="1" and ((r,c) not in visit):
                    bfs(r,c)
                    count+=1
        return count
        

'''
279. Perfect Squares
dynamic programming
So in the DP Array (Dynamic Programming Array) the rest except for the first one will be every big, why?, well because, you need to find the smallest value right? but if it samll like 0 and there is something bigger like x, it will need to be like this min(0,x) well 0 would be smaller and the answer will be 0. the for loop is basically finding all the perfect square what PS stands for, so I did it to make it faster, not anything else, and also the second for loop is calcualting, so we will find lots of possible results using Dynamic Programming and then, compare them. well thats it, you final answer will be at the end.
'''
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [20000 for _ in range(n + 1)]
        dp[0] = 0
        ps = []

        for i in range(1,n + 1):
            if pow(i, 2) > n:
                break
            ps.append(i ** 2)

        for i in range(n + 1):
            for j in ps:
                if i + j <= n:
                    dp[i + j] = min(dp[i] + 1, dp[i + j])
                else:
                    break

        return dp[n]
    
'''
322. Coin Change
'''
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        amount +=1
        liste = [amount] * amount

        liste[0] = 0 

        for coin in sorted(coins):
            for i in range(coin,amount):
                eklenecek = liste[i-coin] + 1
                if eklenecek < liste[i]:
                    liste[i] = eklenecek
        return -1 if liste[-1] == amount else liste[-1]
        
'''
733. Flood Fill
Intuition
The flood fill problem is about changing the color of a connected region of pixels in an image starting from a given pixel. The image is represented as a grid of pixels, where each pixel has a color, and our task is to change the color of all pixels that are connected to the starting pixel (with the same initial color) to a new color.

This is a classic problem that can be solved using graph traversal algorithms, like Depth-First Search (DFS) or Breadth-First Search (BFS). We will use BFS here, which processes pixels level by level, making it well-suited for this type of problem.

Approach
Step-by-Step Explanation
Initial Check:
Before diving into the BFS algorithm, we first check if the color of the starting pixel is the same as the target color. If it is, we immediately return the image as is, because there's no need to modify the image.

BFS Initialization:

We use a queue to implement the BFS algorithm. This queue will store the pixels that we need to process.
We start by adding the initial pixel to the queue.
Immediately, we change the color of the starting pixel to the new color to prevent revisiting it.
BFS Traversal:

The algorithm explores the image in a level-wise manner, processing one pixel and then its neighbors. It continues to expand the traversal to the neighboring pixels until all connected pixels of the same initial color are processed.
In BFS, each pixel is processed once, and all its valid neighbors (i.e., pixels with the same initial color) are added to the queue for further processing.
Exploration of Neighbors:

For each pixel, we check its four neighbors (up, down, left, right). If any of these neighbors have the same color as the starting pixel, they are added to the queue and updated with the new color.
This process continues until the queue is empty, meaning all connected pixels have been processed.
Return the Updated Image:

After the BFS completes, all the connected pixels with the same color as the starting pixel have been filled with the new color. We return the modified image.
'''
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        cur = image[sr][sc]
        if cur == color:
            return image

        n = len(image)
        m = len(image[0])
        q = deque()
        q.append((sr, sc))
        image[sr][sc] = color

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while q:
            row, col = q.popleft()

            for dr, dc in directions:
                newr = dr + row
                newc = dc + col
                if 0 <= newr < n and 0 <= newc < m and image[newr][newc] == cur:
                    q.append((newr, newc))
                    image[newr][newc] = color
        
        return image
    
'''
865. Smallest Subtree with all the Deepest Nodes
Intuition
Assume there are just 3 nodes, A, A.left and A.right. If I want to find the result, my goal would be to be to take the depth of my right and left subtree. If the depth is same and it is the max depth, the current node A is the result. If it is different, the answer is in subtree A.left or subtree A.right.

Approach
Maintain the depth of each node, in a top-down approach. Get the max depth of a nodes left and right subtrees in a bottom-up approach. Simultaneously update the max depth too.

If the left and right depth are equal and are the max_depth, then it means the current node is the parent of these two nodes. This can be considered in our answers domain.

For the leaf nodes, we can check if the leaf node is the deepest till we have reached. If it is, that is one candidate in our answer's domain.
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        ans = None
        max_depth = 0

        def foo(root, d = 0):
            nonlocal max_depth, ans
            if not root:    return 0
            
            max_depth = max(max_depth, d)

            if not root.left and not root.right:
                if d == max_depth:
                    ans = root
                return d
            
            left = foo(root.left, d + 1)
            right = foo(root.right, d + 1)

            if left == right == max_depth:
                ans = root
            print(root.val, left, right, max_depth, d)
            return max(left, right)
        
        foo(root)
        return ans

'''
994. Rotting Oranges
Intuition
Imagine each rotten orange as a source of infection that spreads to adjacent fresh oranges. This spread happens level by level, where all oranges rotting at time t infect their neighbors before time t+1. BFS is ideal for such scenarios because it processes nodes (oranges) level by level.

The problem can be thought of as:

Determining the minimum time required to rot all fresh oranges.
Identifying if any fresh oranges remain uninfected after the spread.
This is a multi-source BFS problem because multiple rotten oranges can simultaneously start spreading the rot.

Approach
We solve the problem step-by-step as follows:

1. Parse the Grid
Identify the positions of all rotten oranges (2) and add them to a queue, marking their starting time as 0.
Count the total number of fresh oranges (1).
Ignore empty cells (0).
2. Perform BFS
Use a queue to process each rotten orange and spread the rot to its 4 adjacent neighbors (up, down, left, right).
If a neighboring cell contains a fresh orange, convert it to rotten, mark it as visited, and add it to the queue with the next time increment.
Track the time taken for the last orange to rot.
3. Check the Final State
If the count of fresh oranges rotted matches the total fresh oranges initially identified, return the time taken.
If not, return -1 because some oranges could not be rotted.

'''

class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        vis = [[0] * m for _ in range(n)]
        q = deque()
        fresh_count = 0

        # Initialize the visited array and queue
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 2:
                    vis[i][j] = 2
                    q.append(((i, j), 0))  # Push rotten orange with time 0
                elif grid[i][j] == 1:
                    fresh_count += 1

        # Define the directions for 4-way movement
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        time = 0
        rotted_count = 0

        # Process the queue
        while q:
            (row, col), t = q.popleft()
            time = max(time, t)  # Track the maximum time

            # Traverse in 4 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < n and 0 <= new_col < m and grid[new_row][new_col] == 1 and vis[new_row][new_col] == 0:
                    q.append(((new_row, new_col), t + 1))
                    vis[new_row][new_col] = 2  # Mark as visited (rotted)
                    rotted_count += 1  # Increment the count of rotted oranges

        # Check if all fresh oranges were rotted
        if rotted_count != fresh_count:
            return -1

        return time
    
'''
1514. Path with Maximum Probability
Intuition
A version of Dijkstra's algorithm. we have to check the shortest path but with min probability, use maxHeap, since python does not have an inbuilt maxHeap we will basically multiply prob with -1.

Approach
make a graph using dictionary, it's a bidirectional graph so add the valus in both node.
take a heap and add a tuple (-1,source) initially. we are taking -1 instead of conventional 0 since we'll be multiplying the probabilities
make a visited set to not overrun the function multiple times.
we'll pop node and probability each time until heap is empty.
if we reach the node, return current cost * (-1) -> remember it's a maxHeap, add value to the visited set.
watch for neighbor nodes in our graph and multiply their probabilities. add them along with the node to the heap for further iterations.
 '''

class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        adj = [[] for _ in range(n)]
        for i, (a, b) in enumerate(edges):
            adj[a].append((b, succProb[i]))
            adj[b].append((a, succProb[i]))
        
        prob = [0] * n
        prob[start] = 1.0
        
        pq = [(-1.0, start)]
        while pq:
            curr_prob, node = heapq.heappop(pq)
            curr_prob = -curr_prob
            
            if node == end:
                return curr_prob
            
            for neighbor, edge_prob in adj[node]:
                new_prob = curr_prob * edge_prob
                if new_prob > prob[neighbor]:
                    prob[neighbor] = new_prob
                    heapq.heappush(pq, (-new_prob, neighbor))
        
        return 0.0