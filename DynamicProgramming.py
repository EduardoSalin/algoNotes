'''
Dynamic Programming - Output depends on computing the task on previous data/simpler problems until simplest base case is reached
Fn = (F(n-1) + F(n-2))
Exponential-time solutions

Knapsack problem: Each item has value v[i] and weight w[i]. Find the combination of items less than <W> pounds that maximizes value
Time complexity: O(numItems * maxWeight). "pseudo-polynomial" time. (but it is on the low end of exponential)
Space complexity: maxWeight (don't need to store previous levels of recursion result)

K(maxWeight, end) = the max value reached when you only consider items i[0] to i[end] and reduced subproblem weight defined by maxWeight
K(maxWeight, end) =>if end == 0 return 0 #(no more items left in sublist)
                    elif maxWeight == 0 return 0 #(knapsack weight all used up)
                    else return max of:
                        #if we include the item 'end'              #if we don't include the item
                         K(maxWeight-w[end], end-1) + v[end] <==> K(maxWeight, end-1)

Unbounded knapsack: allows repeating the same item
Time complexity O(n * W) same as before.
Space complexity: W
K(x,j) = {base cases} else return max of:
                        #if we don't include the item         
                        K(x,j-1)           
                        #if we choose any number of copies of item j (do not reduce the subset of items j if the item is chosen, keep that item in the pool)
                        K(x-w[j], j) + v[j]
                             

Program to figure out how many Perfect Squares needed to sum to a given number n:
dp(n) = min(dp[n-1],dp[n-4],dp[n-9]...for all perfect squares)

Program to figure out least number of coins needed to reach a value n
dp(n) = min(dp[n-coins[1]], dp[n-coins[2],...for all coin values])

Longest common subsequence of letters in two strings X and Y (does not need to be sequential, ex. 13 is a subsequence of 1234)
LCS(i,j): if x[i] = y[i] then return LCS(i-1,j-1) + 1
            else return max of:
                LCS(i-1, j)
                LCS(i, j-1)
'''

#not functional but return the number of unique paths from top left to bottom right corner
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    #dp[0][0] = 1
    #dp[i][j] = dp[i-1][j] + dp[i, j-1]

    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for _ in range(m)] for _ in range(n)]
    dp[0][0] = 1

    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                continue
            if i >= 1:
                dp[i][j] += dp[i-1][j]
            if j >= 1:
                dp[i][j] += dp[i][j-1]
    
    return dp[m-1][n-1]
                

#minimum path sum
def minPathSum(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    dp = grid.copy()

    for i in range(m):
        for j in range(n):
            if i==0 and j == 0:
                continue
            temp = float(inf)
            if i >= 1:
                temp = min(temp, dp[i-1][j])
            if j >= 1:
                temp = min(temp, dp[i][j-1])
            dp[i][j] += temp

            #dp[i][j] = min(dp[i-1][j],dp[i][j-1])
    return dp[m-1][n-1]

'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, 
the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will 
automatically contact the police if two adjacent houses were broken into on the same night.
'''
def rob(self, nums: List[int]) -> int:
    #dp[i] = max profit if you only rob 0,1,2..i
    #dp[i] = if not rob: dp[i-1], if rob: nums[i]+dp[i-2]. Take the maximum
    n = len(nums)
    if n == 1:
        return nums[0]
    dp = [0 for _ in range(n)]
    dp[0] = nums[0]
    dp[1] = max(nums[0],nums[1])
    for i in range(2,n):
        dp[i] = max(dp[i-1], nums[i]+dp[i-2])

    return dp[n-1]

#Knapsack problem to find the least total number of items that make the weight
def dp_make_weight(egg_weights, target_weight):
    least_taken = float("inf")

    if target_weight == 0:
        return 0
    elif target_weight > 0:
        for weight in egg_weights:
            sub_result = dp_make_weight(egg_weights, target_weight - weight)
            least_taken = min(least_taken, sub_result)

    return least_taken + 1
  
if __name__ == "__main__":
    print(dp_make_weight((1, 6, 9), 14))

#Grid solution to the knapsack problem
def knapsack(values, weights, capacity):
    n = len(values)
    # Create a DP table with (n+1) rows and (capacity+1) columns
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Populate the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # Include the item or exclude it
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                # Exclude the item
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find which items were included
    w = capacity
    items_included = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            items_included.append(i - 1)  # Add the item index
            w -= weights[i - 1]

    items_included.reverse()  # Reverse to get the order of selection
    return dp[n][capacity], items_included


# Example usage:
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

max_value, items = knapsack(values, weights, capacity)
print(f"Maximum value: {max_value}")
print(f"Items included: {items}")
