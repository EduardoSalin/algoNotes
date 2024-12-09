1. #0/1 Knapsack Problem Variations
#416. Partition Equal Subset Sum

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]
#2.	494. Target Sum

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if (total - target) % 2 != 0 or total < target:
            return 0
        neg = (total - target) // 2
        dp = [0] * (neg + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(neg, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[neg]
#3.	1049. Last Stone Weight II

class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        target = total // 2
        dp = [0] * (target + 1)
        
        for stone in stones:
            for j in range(target, stone - 1, -1):
                dp[j] = max(dp[j], dp[j - stone] + stone)
        
        return total - 2 * dp[target]
 
#2. Unbounded Knapsack Problem Variations
#4.	322. Coin Change

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
#5.	518. Coin Change 2

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
 
#3. Bounded Knapsack Problem Variations
#	474. Ones and Zeroes

class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for s in strs:
            zeros = s.count('0')
            ones = s.count('1')
            
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        
        return dp[m][n]
#7.	879. Profitable Schemes
#	Type: Bounded Knapsack with additional constraints.

class Solution:
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        MOD = 10**9 + 7
        dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for g, p in zip(group, profit):
            for i in range(n, g - 1, -1):
                for j in range(minProfit, -1, -1):
                    dp[i][j] += dp[i - g][max(0, j - p)]
                    dp[i][j] %= MOD
        
        return sum(dp[i][minProfit] for i in range(n + 1)) % MOD
 
#4. Multi-Dimensional Knapsack Problems
#8.	1099. Two Sum Less Than K
#	Type: Weight Constraint Knapsack.
#	Problem: Find the largest sum less than K for any two numbers.

class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        nums.sort()
        i, j = 0, len(nums) - 1
        max_sum = -1
        
        while i < j:
            total = nums[i] + nums[j]
            if total < k:
                max_sum = max(max_sum, total)
                i += 1
            else:
                j -= 1
        
        return max_sum

#5. 0/1 Knapsack Problems (Continued)
#9.	474. Ones and Zeroes
#	Type: Multidimensional 0/1 Knapsack.

class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeros, ones = s.count('0'), s.count('1')
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        return dp[m][n]
#10.	1458. Max Dot Product of Two Subsequences
#	Type: Subsequence Optimization.

class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        m, n = len(nums1), len(nums2)
        dp = [[-float('inf')] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = max(dp[i][j],
                               dp[i - 1][j],
                               dp[i][j - 1],
                               dp[i - 1][j - 1] + nums1[i - 1] * nums2[j - 1])
        return dp[m][n]
 
#6. Unbounded Knapsack Problems (Continued)
#11.	279. Perfect Squares

class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, int(n ** 0.5) + 1):
            square = i * i
            for j in range(square, n + 1):
                dp[j] = min(dp[j], dp[j - square] + 1)
        return dp[n]
#12.	983. Minimum Cost For Tickets
#	Type: Interval Unbounded Knapsack.

class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dp = [0] * (days[-1] + 1)
        day_set = set(days)
        for i in range(1, days[-1] + 1):
            if i not in day_set:
                dp[i] = dp[i - 1]
            else:
                dp[i] = min(dp[max(0, i - 1)] + costs[0],
                            dp[max(0, i - 7)] + costs[1],
                            dp[max(0, i - 30)] + costs[2])
        return dp[days[-1]]
 
#7. Bounded Knapsack Problems (Continued)
#13.	1235. Maximum Profit in Job Scheduling
#	Type: Weighted Job Scheduling.

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
        dp = [[0, 0]]  # [end_time, profit]
        
        for s, e, p in jobs:
            i = bisect.bisect_right(dp, [s]) - 1
            if dp[i][1] + p > dp[-1][1]:
                dp.append([e, dp[i][1] + p])
        
        return dp[-1][1]
#14.	198. House Robber
#	Type: Subset Sum with adjacency constraint.
#	Problem: Maximize the sum of non-adjacent elements.

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]
#15.	213. House Robber II
#	Type: Circular adjacency constraint.

class Solution:
    def rob(self, nums: List[int]) -> int:
        def rob_helper(nums):
            if not nums:
                return 0
            if len(nums) == 1:
                return nums[0]
            dp = [0] * len(nums)
            dp[0], dp[1] = nums[0], max(nums[0], nums[1])
            for i in range(2, len(nums)):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]
        
        if len(nums) == 1:
            return nums[0]
        return max(rob_helper(nums[:-1]), rob_helper(nums[1:]))
 
#8. Advanced Knapsack Problems
#	188. Best Time to Buy and Sell Stock IV

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices:
            return 0
        if k >= len(prices) // 2:
            return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))
        
        dp = [[0] * len(prices) for _ in range(k + 1)]
        for t in range(1, k + 1):
            max_diff = -prices[0]
            for d in range(1, len(prices)):
                dp[t][d] = max(dp[t][d - 1], prices[d] + max_diff)
                max_diff = max(max_diff, dp[t - 1][d] - prices[d])
        
        return dp[k][-1]
#17.	265. Paint House II
#	Type: Cost Minimization.

class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs:
            return 0
        prev_min, prev_second, prev_idx = 0, 0, -1
        for row in costs:
            curr_min, curr_second, curr_idx = float('inf'), float('inf'), -1
            for i, cost in enumerate(row):
                if i == prev_idx:
                    cost += prev_second
                else:
                    cost += prev_min
                if cost < curr_min:
                    curr_second, curr_min = curr_min, cost
                    curr_idx = i
                elif cost < curr_second:
                    curr_second = cost
            prev_min, prev_second, prev_idx = curr_min, curr_second, curr_idx
        return prev_min

#1. LeetCode 416: Partition Equal Subset Sum (Medium)
#Problem: Given an array nums, determine if it can be partitioned into two subsets such that the sums of both subsets are equal.

def canPartition(nums):
    total_sum = sum(nums)  # Calculate the total sum of the array
    if total_sum % 2 != 0:  # If the total sum is odd, we cannot split it into two equal parts
        return False

    target = total_sum // 2  # Target sum for each subset
    dp = [False] * (target + 1)  # DP array to store possible subset sums
    dp[0] = True  # Base case: Sum of 0 is always achievable (choose no elements)

    for num in nums:  # Iterate through each number in the array
        for t in range(target, num - 1, -1):  # Iterate backwards to avoid overwriting dp values
            dp[t] = dp[t] or dp[t - num]  # Update dp[t] if we can form it by adding `num`
    
    return dp[target]  # The result is whether the target sum is achievable
#2. LeetCode 494: Target Sum (Medium)
#Problem: You are given an integer array nums and a target target. Assign + or - signs to each element in nums to calculate the target. Return the number of ways to do this.

def findTargetSumWays(nums, target):
    total_sum = sum(nums)  # Calculate the total sum of the array
    if abs(target) > total_sum or (total_sum + target) % 2 != 0:  # Check for invalid cases
        return 0

    subset_sum = (total_sum + target) // 2  # Transform into a subset sum problem
    dp = [0] * (subset_sum + 1)  # DP array to count subset sums
    dp[0] = 1  # Base case: There is one way to make a sum of 0 (choose no elements)

    for num in nums:  # Iterate through each number
        for t in range(subset_sum, num - 1, -1):  # Iterate backwards
            dp[t] += dp[t - num]  # Update dp[t] by adding ways to form (t - num)
    
    return dp[subset_sum]  # Return the number of ways to form `subset_sum`
#3. LeetCode 322: Coin Change (Medium)
#Problem: Given an integer array coins representing denominations and an integer amount, return the fewest number of coins that make up the amount. If it’s not possible, return -1.

def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)  # Initialize DP array with "infinity"
    dp[0] = 0  # Base case: 0 coins are needed to make a sum of 0

    for coin in coins:  # Iterate through each coin denomination
        for t in range(coin, amount + 1):  # Iterate from the coin value up to the amount
            dp[t] = min(dp[t], dp[t - coin] + 1)  # Update dp[t] with the minimum coins needed
    
    return dp[amount] if dp[amount] != float('inf') else -1  # Return result
#Explanation:
#1.	dp = [float('inf')] * (amount + 1): Initialize a DP array where dp[t] represents the minimum number of coins needed to form sum t.
#2.	dp[0] = 0: Base case: 0 coins are required to make a sum of 0.
#3.	for coin in coins:: Iterate through each coin denomination.
