## Invalid Transactions

Because the limit allows O(N^2) to pass, we choose the most straightforward solution where we just look at each pair of transactions.

For each transaction, we either have the amount >= 1000, or there exists another transaction satisfying the condition.

(Note that there are solutions with better complexities - I encourage you to explore these too.)

~~~python 
class Solution(object):
    def invalidTransactions(self, transactions):
        NAME, TIME, AMOUNT, CITY = range(4)
        
        txrows = []
        for row in transactions:
            name, time, amount, city = row.split(',')
            time = int(time)
            amount = int(amount)
            txrows.append((name, time, amount, city))
        
        ans = []
        for i, tx1 in enumerate(txrows):
            if tx1[AMOUNT] >= 1000:
                ans.append(transactions[i])
                continue
            for j, tx2 in enumerate(txrows):
                if (i != j and abs(tx1[TIME] - tx2[TIME]) <= 60 and
                        tx1[NAME] == tx2[NAME] and tx1[CITY] != tx2[CITY]):
                    ans.append(transactions[i])
                    break
        
        return ans
~~~


Compare Strings by Frequency of the Smallest Character

Obviously, we need to be able to calculate f(word) for each word in words/queries, where f is as in the problem statement.  The answer is just f(word) = word.count(min(word)).

Now we can directly calculate the sum.  We could speed this up by a constant factor by precalculating sum(count[i:]) first, but its not necessary.

~~~python 
class Solution(object):
    def numSmallerByFrequency(self, queries, words):
        def smallfreq(s):
            return s.count(min(s))
        
        count = [0] * 11
        for word in words:
            count[smallfreq(word)] += 1
        
        ans = []
        for query in queries:
            f = smallfreq(query)
            ans.append(sum(count[f+1:]))
        return ans
~~~
Remove Zero Sum Consecutive Nodes from Linked List

For convenience, let's think of the linked list as an array A of its values.

Clearly, there's a brute force solution idea:  for each i=0..A.length-1, find the first j >= i so that sum(A[i..j]) is zero, and delete it.  (Here, A[i..j] refers to the subarray A[i], ..., A[j].)

Does this work?  Suppose after these deletions, there is a zero sum subarray that was originally A[i..j] minus some deletions.  All those deletions are sum zero, so A[i..j] would have been sum zero and deleted first, a contradiction.

Now, how can we make it fast?  For each index j, let's find the smallest index i with the same prefix sum.  (This is what the while loop does.)  Then, at the node i (prev), we will make it jump over this position (prev.next = cur.next).

~~~python 
class Solution(object):
    def removeZeroSumSublists(self, head):
        sentinel = cur = ListNode(0)
        sentinel.next = head
        prefix_sum = 0
        seen = collections.OrderedDict()
        
        while cur:
            prefix_sum += cur.val
            prev = cur
            while prefix_sum in seen:
                prev = seen.popitem()[1]
            seen[prefix_sum] = prev
            prev.next = cur = cur.next
        return sentinel.next
~~~
Dinner Plate Stacks

If this question only had push and pop, it would be trivial, so we focus on how to speed up the function popAtStack.  popAtStack basically creates "holes" where when we push, we prioritize a hole over the normal pushing at the end.

Because we want to repeatedly push to the smallest index hole (ie., repeatedly query the min of a set), a heap is a natural choice.

This creates our psuedocode:

push:
  Try to push to each hole listed in the heap, as long as its smaller than the usual operation
  If you can't, push to the end
pop:
  Pop from the end
popAtStack:
  Pop from that stack, creating a new hole
  
~~~python   
class DinnerPlates(object):

    def __init__(self, capacity):
        self.A = []
        self.cap = capacity
        self.open = []  # heap of indexes where holes exist
        
    def push(self, x):
        while self.open:
            i = heapq.heappop(self.open)
            if i < len(self.A) and len(self.A[i]) < self.cap:
                self.A[i].append(x)
                return
        
        if self.A and len(self.A[-1]) < self.cap:
            self.A[-1].append(x)
        else:
            self.A.append([x])

    def pop(self):
        while self.A and not self.A[-1]:
            self.A.pop()
        if not self.A:
            return -1
        return self.A[-1].pop()

    def popAtStack(self, i):
        if i < len(self.A) and self.A[i]:
            heapq.heappush(self.open, i)
            return self.A[i].pop()
        return -1
~~~
amit_sharma — 08/25/2019
"if cursum in prefixes:
                prefixes[cursum].next = cur.next"
I cannot understand this lines.. cursum is used as index of prefixes... so "if cursum in prefixes:" is like searching index in the prefix array which will be always true.. or is it some dedicated python trick??
Alex Wice — 08/25/2019
prefixes is a dictionary from a prefix sum to a node.   "cursum in prefixes" means is cursum a key of the dictionary prefixes, ie.  is cursum a prefix sum we have seen before?
Lareine — 08/26/2019
What algorithm is this?
~~~python 
class MinimumSpanningTree:
    def __init__(self, n):
        r = range(1, n + 1)
        self.forest, vertexIndex = [{house} for house in r], {house: house - 1 for house in r}
    def foobar(self, pipes):
        for house1, house2, cost in pipes:
            house1, house2 = sorted([house1, house2], key = lambda x: self.vertexIndex[x])
            self.forest[self.vertexIndex[house1]].add(house2)
            self.forest[self.vertexIndex[house2]].remove(house2)
            self.vertexIndex[house2] = self.vertexIndex[house1]
        self.forest = [tree for tree in self.forest if tree]
        for i in range(len(self.forest)):
            for house in self.forest[i]: self.vertexIndex[house] = i
~~~
Alex Wice — 08/26/2019
I don't know what you mean @Lareine , I understand what the code is trying to do though [but i think its incorrect]
but without seeing the problem I cant say for sure
Lareine — 08/28/2019
This is a problem from Saturday's biweekly contest https://leetcode.com/contest/biweekly-contest-7/problems/optimize-water-distribution-in-a-village/
Level up your coding skills and quickly land a job. This is the best place to expand your knowledge and get prepared for your next interview.
~~~python 
class Tree:
    def __init__(self, house): self.i, self.house, self.well = len(forest), house, house[0] if len(house) == 1 else -1
def minCostToSupplyWater(n, wells, pipes):
    def put(tree, vertex):
        forest[houseIndex[tree]].house.append(vertex)
        houseIndex[vertex] = houseIndex[tree]
    r, forest = range(1, n + 1), []
    houseIndex, pipeCost = {house: -1 for house in r}, 0
    for house1, house2, cost in sorted([[house, None, wells[house - 1]] for house in r] + pipes, key = lambda x: x[2]):
        if house2:
            if houseIndex[house1] == -1:
                if houseIndex[house2] == -1:
                    houseIndex[house1] = houseIndex[house2] = len(forest)
                    forest.append(Tree([house1, house2]))
                else: put(house2, house1)
                pipeCost += cost
            elif houseIndex[house2] == -1:
                put(house1, house2)
                pipeCost += cost
            elif forest[houseIndex[house1]].well == -1 or forest[houseIndex[house2]].well == -1:
                source, destination = sorted([house1, house2], key = lambda x: len(forest[houseIndex[x]].house))
                i, j = houseIndex[source], houseIndex[destination]
                if i != j:
                    forest[j].house += forest[i].house
                    for h in forest[i].house: houseIndex[h] = j
                    forest[i].house = []
                    forest[i].well, forest[j].well = sorted([forest[i].well, forest[j].well])
                    pipeCost += cost
        elif houseIndex[house1] == -1:
            houseIndex[house1] = len(forest)
            forest.append(Tree([house1]))
        elif forest[houseIndex[house1]].well == -1: forest[houseIndex[house1]].well = house1
    return sum(wells[tr.well - 1] for tr in forest if tr.well != -1) + pipeCost
~~~
This was my accepted code
1) I think the code quality is really bad - how would you improve it?
2) What algorithm is this called?
Alex Wice — 08/28/2019
1.  Follow pep8 guidelines basically, and have logical structure to your code
2.  It seems like a variant of https://en.wikipedia.org/wiki/Disjoint-set_data_structure  combined with prim's algorithm
In this question, your goal is to get all the nodes connected with n edges, where "ground water" is a n+1-th node.  Then, its a straightforward application of prim's algorithm.
Lareine — 08/28/2019
So the algorithm I wrote wasn't Kruskal's?
Alex Wice — 08/28/2019
sorry, its kruskal's.  (i mentally consider it the same thing, even though its not)
Lareine — 08/28/2019
Oh, ok
Alex Wice — 08/28/2019
and we should use kruskal's too.
Lareine — 08/28/2019
What's the difference between Prim's and Kruskal's?
Alex Wice — 08/28/2019
~~~python 
class DSU:
    def __init__(self, N):
        self.par = range(N)

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return False
        self.par[yr] = xr
        return True

def minCostToSupplyWater(n, wells, pipes):
    # wells[i] connects node i to node n
    edges = pipes + [[i, n, w] for i, w in enumerate(wells)]
    edges.sort(key = lambda u, v, w: w)

    dsu = DSU(n+1)
    ans = 0
    for u, v, w in edges:
        if dsu.union(u, v):
            ans += w
    return ans
~~~
kruskal: for each edge in weight order, try to add it.
prims:  start with a component = some node, now repeatedly add smallest edge that connects from this component to not-this component
what i would suggest is to run your code through yapf, or various linters like flake8, and do the things it suggests  and try to make your code more like that
the most important one tip is uhh, to not have oneliner syndrome, like always break up the line
(keep in mind the DSU should also have union by rank for optimal complexity, which the code snippet above doesnt have)
Lareine — 08/28/2019
Is my code optimal regarding runtime
Alex Wice — 08/28/2019
It's pretty hard to read your code, let me uhh, lint and rewrite it first
btw the linting thing is not a light point, it seems that most people i meet that write code like this, fail the onsite even if they 'got the right answer'
Alex Wice — 08/28/2019
Ok, its not the optimal complexity because the way you handle the "DSU" parts is to copy out the contents of one forest into another
ie. you do a sort of traversal every time, instead of using path compression
~~~python 
class Tree:
    def __init__(self, group):
        self.i = len(forest)
        self.group = group
        self.well = group[0] if len(group) == 1 else -1


def minCostToSupplyWater(n, wells, pipes):
    def put(tree, v):
        forest[index[tree]].group.append(v)
        index[v] = index[tree]

    # leader = house: representative in connected component 
    leader = {h: -1 for h in range(1, n+1)}  
    forest = []
    ans = 0
    
    edges = pipes + [[h, None, wells[h-1]] for h in range(1, n+1)]
    edges.sort(key = lambda (u, v, w): w)
    for u, v, cost in edges:
        if v is not None:
            if leader[u] == leader[v] == -1:
                leader[u] = leader[v] = len(forest)
                forest.append(Tree([u, v]))
                ans += cost
            elif leader[u] == -1:
                put(v, u)
                ans += cost
            elif leader[v] == -1:
                put(u, v)
                ans += cost
            elif forest[index[u]].well == -1 or forest[index[v]].well == -1:
                u2, v2 = sorted([u, v], 
                    key = lambda i: len(forest[index[i]].group))
                lu, lv = leader[u2], leader[v2]
                if lu != lv:
                    forest[lv].group.extend(forest[lu].group)
                    for house in forest[lu].group:
                        index[house] = lv
                    forest[lu].group[:] = []
                    if forest[lu].well > forest[lv].well:
                        forest[lu].well, forest[lv].well = \
                            forest[lv].well, forest[lu].well
                    
                    ans += cost
        elif index[u] == -1:
            index[u] = len(forest)
            forest.append(Tree([u]))
        elif forest[index[u]].well == -1:
            forest[index[u]].well = u
    
    ans += sum(wells[tree.well - 1]
               for tree in forest if tree.well != -1)
    return ans
~~~
you can see in the uhh, "if lu != lv" part of your code how i rewrote it
Lareine — 08/29/2019
I like your DSU class better
Lareine — 09/01/2019
https://leetcode.com/problems/can-make-palindrome-from-substring/
Python O(n + m) solution takes around 4 seconds to run in my IDE but gets TLE in LeetCode IDE
Level up your coding skills and quickly land a job. This is the best place to expand your knowledge and get prepared for your next interview.

Alex Wice — 09/01/2019
Here are my solutions for the most recent contest.

prime arrangements:
~~~python 
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        def isprime(n):
            if n < 2:
                return False
            d = 2
            while d * d <= n:
                if n % d == 0:
                    return False
                d += 1 + (d & 1)
            return True
        
        MOD = 10 ** 9 + 7
        prime_count = sum(isprime(x) for x in range(n+1))

        ans = 1
        for i in range(1, prime_count + 1):
            ans = ans * i % MOD
        for i in range(1, n - prime_count + 1):
            ans = ans * i % MOD
        return ans
~~~


diet plan performance:
~~~python 
class Solution:
    def dietPlanPerformance(self, calories, k, lower, upper):
        eaten = sum(calories[:k-1])
        ans = 0
        for i in range(k-1, len(calories)):
            eaten += calories[i]
            if i >= k: eaten -= calories[i - k]
            if eaten < lower:
                ans -= 1
            if eaten > upper:
                ans += 1
        return ans
~~~
palindrome queries:
~~~python 
class Solution:
    def canMakePaliQueries(self, s, queries):
        P = [0]
        for c in s:
            x = 1 << (ord(c) - ord('a'))
            P.append(P[-1] ^ x)
        
        def possible(query):
            left, right, k = query
            count = P[right + 1] ^ P[left]
            popcount = bin(count).count('1')
            return popcount // 2 <= k

        return map(possible, queries)

~~~
num valid words:
~~~python 
class Solution:
    def findNumOfValidWords(self, words, puzzles):
        count = collections.Counter()
        for word in words:
            code = 0
            for letter in word:
                code |= 1 << (ord(letter) - ord('a'))
            count[code] += 1
        
        def query(puzzle):
            # Query every subset that includes puzzle[0]
            masks = [1 << (ord(puzzle[0]) - ord('a'))]
            for letter in puzzle[1:]:
                for i in range(len(masks)):
                    masks.append(masks[i] | (1 << (ord(letter) - ord('a'))))
            
            return sum(count[mask] for mask in masks)

        return map(query, puzzles)
~~~
Alex Wice — 09/08/2019
# Count Substrings with Only One Distinct Letter
Partition the string into groups that all have the same letter, and adjacent groups have different letters.
For example, "aaabbbaaacccc" would partition to "aaa", "bbb", "aaa", "cccc".  For each group, we can count the number of substrings as length * (length + 1) // 2.

~~~python 
# Soln 1: using itertools.groupby

def countLetters(self, S):
    ans = 0
    for k, grp in itertools.groupby(S):
        length = len(list(grp))
        ans += length * (length + 1) // 2
    return ans


# Soln 2: count groups manually
def countLetters(self, S):
    ans = 0
    anchor = 0
    for i in range(len(S)):
        if i == len(S) - 1 or S[i] != S[i+1]:
            length = i - anchor + 1
            ans += length * (length + 1) // 2
            anchor = i + 1
    return ans
~~~

# Before and After Puzzle

Brute force - for each pair of puzzles, lets check if they form a before and after puzzle.
~~~python 
class Solution(object):
    def beforeAndAfterPuzzles(self, phrases):
        ans = set()
        for i, p1 in enumerate(phrases):
            a1 = p1.split()
            for j, p2 in enumerate(phrases):
                a2 = p2.split()
                if i != j and a1[-1] == a2[0]:
                    ans.add(" ".join(a1 + a2[1:]))
        
        return sorted(ans)
~~~
#Shortest Distance to Target Color
Next array.
Let's compute lefts[color][i] = number of left steps to reach the color from i,
and rights[color][i] = number of right steps to reach the color for i.
Then the answer of each query is min(lefts[color][i], rights[color][i]).

~~~python 
def shortestDistanceColor(self, colors, queries):
    N = len(colors)

    lefts = [None] * N
    steps = [N, N, N]
    for i in range(N):
        for j in range(3):
            steps[j] += 1
        steps[colors[i]] = 0
        lefts[i] = steps[:]

    rights = [None] * N
    steps = [N, N, N]
    for i in range(N):
        for j in range(3):
            steps[j] += 1
        steps[colors[i]] = 0
        rights[i] = steps[:]

    def solve(query):
        i, c = query
        ans = min(lefts[c-1][i], rights[c-1][i])
        return ans if ans < N else -1

    return map(solve, queries)
~~~
Alex Wice — 09/08/2019
#Distance between busstops

One route must use some subarray sum, the other route uses the other numbers.
~~~python 
class Solution:
    def distanceBetweenBusStops(self, distance, start, destination):
        if start > destination:
            start, destination = destination, start
        x = sum(distance[start: destination])
        return min(x, sum(distance) - x)
~~~


# Max subarray sum with one deletion

Next array.  dp0[i] is largest sum that ends at A[i], dp1[i] the largest sum starting at A[i].

~~~python 
class Solution(object):
    def maximumSum(self, A):
        N = len(A)
        dp0 = [None] * N  # largest ending here
        dp1 = [None] * N  # largest starting here
        
        cur = A[0]
        dp0[0] = cur
        for i in xrange(1, N):
            cur = max(cur + A[i], A[i])
            dp0[i] = cur
        
        cur = A[-1]
        dp1[-1] = cur
        for i in xrange(N-2, -1, -1):
            cur = max(cur + A[i], A[i])
            dp1[i] = cur
        
        ans = max(dp0)
        for i, x in enumerate(A):
            if i+2 < N:
                ans = max(ans, dp0[i] + dp1[i+2])
        return ans
~~~
bobbyDrake — 09/08/2019
@Alex Wice is “next array” the name of a technique?
Alex Wice — 09/08/2019
Basically, yeah.   See two of the posts here where I mention it
Make array increasing
There are more clever methods (see forum) but I think this is one of the most straightforward ways
Let dp[i][j] = ....  (as written in the code).  This is natural since every value A[i] can only be itself or some choice from avail
Also, when choosing a value for A[i] to satisfy A[i] > A[i-1], obviously smaller is strictly better.

~~~python 
from functools import lru_cache

class Solution:
    def makeArrayIncreasing(self, A, avail):
        INF = 1 << 30
        avail = sorted(set(avail))
        
        @lru_cache(None)
        def dp(i, cur):
            if i >= len(A):
                return 0
            
            j = bisect.bisect(avail, cur)
            swap = 1 + dp(i+1, avail[j]) if j < len(avail) else INF
            keep = dp(i+1, A[i]) if A[i] > cur else INF
            return min(swap, keep)
        
        ans = dp(0, -INF)
        return ans if ans < INF else -1
~~~
Alex Wice — 09/15/2019
Maximum Number of Balloons
~~~python 
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        count = collections.Counter(text)
        return min(count['b'], count['a'],
                   count['l'] // 2, count['o'] // 2, count['n'])
~~~

Reverse Substrings Between Each Pair of Parentheses
~~~python 
class Solution:
    def reverseParentheses(self, s: str) -> str:
        stack = [[]]
        for c in s:
            if c == '(':
                stack.append([])
            elif c == ')':
                row = stack.pop()
                stack[-1].extend(row)
            else:
                stack[-1].append(c)

        return "".join(stack.pop())
~~~

K-Concatenation Maximum Sum
~~~python 
class Solution(object):
    def kConcatenationMaxSum(self, A, K):
        def kadane(A):
            ans = cur = 0
            for x in A:
                cur = max(cur + x, x)
                ans = max(ans, cur)
            return ans
        
        MOD = 10**9 + 7
        S = sum(A)
        if K == 1:
            return kadane(A) % MOD
        elif S <= 0:
            return kadane(A + A) % MOD
        else:
            ans = (K - 2) * S
            ans += max(itertools.accumulate(A))
            ans += max(itertools.accumulate(reversed(A)))
            return ans % MOD
~~~
Alex Wice — 09/22/2019
How Many Apples Can You Put into the Basket
~~~python 
class Solution(object):
    def maxNumberOfApples(self, A):
        A.sort()
        space = 5000
        ans = 0
        for x in A:
            if space >= x:
                space -= x
                ans += 1
        return ans
~~~


Minimum Knight Moves
~~~python 
class Solution:
    dist = [[None] * 305 for _ in range(305)]
    dist[0][0] = 0
    
    def neighbors(x, y):
        for dx, dy in ((1, 2), (1, -2), (-1, 2), (-1, -2),
                       (2, 1), (2, -1), (-2, 1), (-2, -1)):
            yield abs(x + dx), abs(y + dy)

    queue = collections.deque([[0, 0, 0]])
    while queue:
        x, y, d = queue.popleft()
        for nx, ny in neighbors(x, y):
            if nx < 305 and ny < 305 and dist[nx][ny] is None:
                dist[nx][ny] = d + 1
                queue.append([nx, ny, d + 1])

    def minKnightMoves(self, x, y):
        return Solution.dist[abs(x)][abs(y)]
~~~

Find Smallest Common Element in All Rows
~~~python 
class Solution(object):
    def smallestCommonElement(self, A):
        intersection = reduce(operator.and_, map(set, A))
        return min(intersection) if intersection else -1


class Solution(object):
    def smallestCommonElement(self, A):
        N, C = len(A), len(A[0])
        heads = [0] * N

        for x in A[0]:
            for i in xrange(1, N):
                while heads[i] < C and A[i][heads[i]] < x:
                    heads[i] += 1
                if heads[i] == C or A[i][heads[i]] > x:
                    break
            else:
                return x
        return -1
~~~

Minimum Time to Build Blocks
~~~python 
class Solution(object):
    def minBuildTime(self, A, split_cost):
        heapq.heapify(A)
        while len(A) > 1:
            x = heapq.heappop(A)
            y = heapq.heappop(A)
            heapq.heappush(A, max(x, y) + split_cost)
        return A.pop()
~~~
Alex Wice — 09/22/2019
Minimum Absolute Difference
~~~python 
class Solution(object):
    def minimumAbsDifference(self, A):
        m = min(A[i+1] - A[i] for i in xrange(len(A) - 1))
        
        ans = []
        for i in xrange(len(A) - 1):
            if A[i+1] - A[i] == m:
                ans.append([A[i], A[i+1]])
        return ans
~~~

Ugly Number III
class Solution(object):
    def nthUglyNumber(self, n, a, b, c):
        from fractions import gcd

        def f(x):
            ans = x / a + x / b + x / c
            ans -= x / (a * b / gcd(a, b))
            ans -= x / (b * c / gcd(b, c))
            ans -= x / (c * a / gcd(c, a))
            ans += x / (a * b * c / gcd(a * b, c * gcd(a, b)))
            return ans

        lo, hi = 0, 2 * 10 ** 9
        while lo < hi:
            mi = (lo + hi) / 2
            if f(mi) < n:
                lo = mi + 1
            else:
                hi = mi
        return lo


Smallest String With Swaps
class Solution(object):
    def smallestStringWithSwaps(self, S, pairs):
        N = len(S)
        graph = [[] for _ in xrange(N)]
        for u, v in pairs:
            graph[u].append(v)
            graph[v].append(u)
        ans = [None] * N
        
        seen = [False] * N
        for u in xrange(N):
            if not seen[u]:
                seen[u] = True
                stack = [u]
                component = []
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for nei in graph[node]:
                        if not seen[nei]:
                            seen[nei] = True
                            stack.append(nei)
                
                component.sort()
                letters = sorted(S[i] for i in component)
                for ix, i in enumerate(component):
                    letter = letters[ix]
                    ans[i] = letter
        return "".join(ans)
Smallest String With Swaps (dsu solution)
class DSU:
    def __init__(self, N):
        self.par = range(N)
        self.rnk = [0] * N
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rnk[xr] < self.rnk[yr]:
            xr, yr = yr, xr
        if self.rnk[xr] == self.rnk[yr]:
            self.rnk[xr] += 1

        self.par[yr] = xr

class Solution(object):
    def smallestStringWithSwaps(self, S, pairs):
        N = len(S)
        dsu = DSU(N)
        for u, v in pairs:
            dsu.union(u, v)
        
        components = collections.defaultdict(list)
        for i in xrange(N):
            components[dsu.find(i)].append(i)
        
        ans = [None] * N
        for component in components.values():
            component.sort(key=S.__getitem__)
            for ix, i in zip(component, sorted(component)):
                ans[i] = S[ix]
        return "".join(ans)

Number of Ways to Form a Target String Given a Dictionary
DP.  Let dp(i, j) be the number of ways to write target[i:] with indexes >= j.  We either use a letter from index j (and there are counts[j][target[i]] such choices), or we don't.

from functools import lru_cache

class Solution:
    def numWays(self, A, target):
        MOD = 10 ** 9 + 7
        N = len(target)
        K = len(A[0])
        counts = [collections.Counter() for _ in range(K)]
        for word in A:
            for i in range(K):
                counts[i][word[i]] += 1
        
        @lru_cache(None)
        def dp(i, j):
            if i == N:
                return 1
            if j == K:
                return 0
            
            ans = dp(i, j+1) + counts[j][target[i]] * dp(i+1, j+1)
            return ans % MOD
        
        return dp(0, 0)
Alex Wice — 11/01/2020
Count Sorted Vowel Strings

DP.  At time t, let dp[i] be the number of strings of length t that end in vowel i, and ndp[i] the number of strings of length t + 1 ending in vowel i.  Each ndp[j] can be formed by adding vowel j onto any string counted by dp[i] with i <= j.

class Solution:
    def countVowelStrings(self, n):
        dp = [1] * 5
        for _ in range(n - 1):
            ndp = [0] * 5
            for j in range(5):
                for i in range(j + 1):
                    ndp[j] += dp[i]
            dp = ndp
        
        return sum(dp)
Alex Wice — 11/02/2020
Furthest Building You Can Reach

If you can reach index i, you can reach any index j < i.  Thus, we can binary search for the answer and solve the decision problem instead.

Look at pairs of adjacent buildings between H[..target].  We use ladders on the top ladders of them, and the rest need bricks.

class Solution:
    def furthestBuilding(self, H, bricks, ladders):
        def possible(target):
            path_heights = sorted(max(0, H[i+1] - H[i]) for i in range(target))
            return sum(path_heights[:-ladders or None]) <= bricks
        
        lo, hi = 0, len(H) - 1
        while lo < hi:
            mi = lo + hi + 1 >> 1
            if possible(mi):
                lo = mi
            else:
                hi = mi - 1
        return lo
Alternative Solution

Go left to right, storing the ladders largest deltas.  The rest of the deltas must be covered by bricks.

from heapq import heappush, heappop

class Solution:
    def furthestBuilding(self, H, bricks, ladders):
        pq = []  # store `ladders` largest deltas
        for i in range(len(H) - 1):
            delta = H[i + 1] - H[i]
            if delta > 0:
                heappush(pq, delta)
                if len(pq) > ladders:
                    bricks -= heappop(pq)
                    if bricks < 0:
                        return i
        
        return len(H) - 1
Alex Wice — 11/02/2020
Kth Smallest Instructions

Say there are h horizontal moves and v vertical moves left to do.

If we put an instruction like H, there will be ways = binom(h - 1 + v, v) ways to arrange the rest of the instructions.  Therefore, if k <= ways, we know the k-th instruction will be within the first ways of them and therefore start with 'H'.

class Solution:
    def kthSmallestPath(self, destination, k):
        v, h = destination
        
        ans = []
        while h or v:
            ways = math.comb(h - 1 + v, v)
            if k <= ways:
                ans.append('H')
                h -= 1
            else:
                ans.append('V')
                v -= 1
                k -= ways
        
        return "".join(ans)
Alex Wice — 11/09/2020
Get Maximum in Generated Array

Create the array from left to right.

class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        arr = [0] * (n + 1)
        if n >= 1:
            arr[1] = 1

        for i in range(2, n + 1):
            if i % 2 == 0:
                arr[i] = arr[i >> 1]
            else:
                arr[i] = arr[i >> 1] + arr[i + 1 >> 1]
            
        return max(arr)
Minimum Deletions to Make Character Frequencies Unique

We only care about the frequencies of the letters.  Also, if there are say 1000 "a"'s (and all other counts are less), we can ignore all the "a"'s.

So we just have to make the (sorted descending) frequencies strictly decreasing, greedily subtracting as we go.

class Solution:
    def minDeletions(self, s: str) -> int:
        freqs = list(collections.Counter(s).values())
        freqs.sort(reverse=True)
        
        # eg. freqs = [4, 4, 3, 3, 2]
        #     want    [4, 3, 2, 1, 0]
        ans = 0
        for i in range(1, len(freqs)):
            if freqs[i] >= freqs[i - 1]:
                target = max(0, freqs[i - 1] - 1)
                ans += freqs[i] - target
                freqs[i] = target
        
        return ans
Sell Diminishing-Valued Colored Balls

There is some largest possible cutoff t for which selling down to max(x, t) balls for each color is going to sell at least X balls.  We can find this by binary search.  After, we have to refund sold - X balls at a price of t + 1.

class Solution:
    def maxProfit(self, A, X):
        # Want to sell X balls
        
        def sum_range(lo, hi):
            return hi * (hi + 1) // 2 - lo * (lo - 1) // 2

        def num_sold(t):
            return sum(max(0, x - t) for x in A)

        lo, hi = 0, max(A)  # Want largest t with num_sold(t) >= X
        while lo < hi:
            mi = lo + hi + 1 >> 1
            if num_sold(mi) >= X:
                lo = mi
            else:
                hi = mi - 1

        t = lo
        sold = num_sold(t)
        profit = 0
        for x in A:
            if x > t:
                profit += sum_range(t + 1, x)
        
        profit -= (sold - X) * (t + 1)
        
        return profit % (10 ** 9 + 7)
Alex Wice — 11/09/2020
Create Sorted Array through Instructions

To know how many elements are less than the next instruction x, we use a sorted list (balanced tree datastructure) and binary search on it.

from sortedcontainers import SortedList

class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        A = SortedList()
        ans = 0

        for x in instructions:
            less = A.bisect_left(x)
            greater = len(A) - A.bisect_right(x)
            ans += min(less, greater)
            A.add(x)
        
        return ans % (10 ** 9 + 7)
Alex Wice — 11/15/2020
Defuse the Bomb

Brute force.  You can use a pointer j modulo N to keep track of the position of the addend in a circular array.

For the linear solution, use prefix sums.  The hard part is when the intended sum A[lo..hi] doesn't fit in the array - it needs to be broken into two intervals.

class Solution:
    def decrypt(self, A, K):
        N = len(A)
        ans = [0] * N
        
        if K > 0:
            for i in range(N):
                j = i
                for _ in range(K):
                    j += 1
                    ans[i] += A[j % N]
        elif K < 0:
            for i in range(N):
                j = i
                for _ in range(-K):
                    j -= 1
                    ans[i] += A[j % N]
        
        return ans
class Solution:
    def decrypt(self, A, K):
        N = len(A)
        P = [0]
        for x in A:
            P.append(P[-1] + x)
        
        def sum_range(lo, hi):
            # sum of A[lo..hi]
            return P[hi + 1] - P[lo]
        
        ans = [0] * N
        if K > 0:
            for i in range(N):
                lo, hi = i + 1, i + K
                if hi >= N:
                    ans[i] = sum_range(lo, N - 1) + sum_range(0, hi % N)
                else:
                    ans[i] = sum_range(lo, hi)
        elif K < 0:
            for i in range(N):
                lo, hi = i + K, i - 1
                if lo < 0:
                    ans[i] = sum_range(0, hi) + sum_range(lo % N, N - 1)
                else:
                    ans[i] = sum_range(lo, hi)
        
        return ans
Minimum Deletions to Make String Balanced

Sweepline.  Say there is some vertical line for which the left part is all "a"'s and the right part is all "b"'s.  We will need to delete the left "b"'s and the right "a"'s to make the string balanced.

class Solution:
    def minimumDeletions(self, s: str) -> int:
        lb = 0
        ans = ra = s.count('a')
        
        for c in s:
            if c == 'a':
                ra -= 1
            else:
                lb += 1
            
            ans = min(ans, lb + ra)
        
        return ans
Minimum Jumps to Reach Home

Since it's a shortest path, we think of BFS.  The states are (location, can_back).  The only thing we need now is to write the neighbors function.

class Solution:
    def minimumJumps(self, forbidden, d_fow, d_back, target) -> int:
        banned_set = set(forbidden)
        
        def neighbors(node):
            loc, can_back = node
            nloc = loc + d_fow
            if nloc <= 6000 and nloc not in banned_set:
                yield nloc, True
            
            if can_back:
                nloc = loc - d_back
                if nloc >= 0 and nloc not in banned_set:
                    yield nloc, False

        queue = [(0, True)]
        dist = {(0, True): 0}
        
        for node in queue:
            if node[0] == target:
                return dist[node]
            for nei in neighbors(node):
                if nei not in dist:
                    dist[nei] = 1 + dist[node]
                    queue.append(nei)
        
        return -1
Distribute Repeating Integers

We only care about the frequencies of nums.  Now mask dp: (i, mask) represents we are on the i-th product-frequency, and we have mask remaining customers to satisfy.

class Solution:
    def canDistribute(self, nums: List[int], customers: List[int]) -> bool:
        freqs = list(collections.Counter(nums).values())
        totals = [0] * (1 << len(customers))
        for mask in range(1 << len(customers)):
            for bit in range(len(customers)):
                if mask >> bit & 1:
                    totals[mask] += customers[bit]

        @lru_cache(None)
        def dp(i, rem):
            if rem == 0:
                return True
            if i == len(freqs):
                return False
            
            submask = rem
            while submask:
                demand = totals[submask]
                if freqs[i] >= demand:
                    if dp(i + 1, rem ^ submask):
                        return True
                submask = (submask - 1) & rem
            
            return dp(i + 1, rem)
        
        return dp(0, (1 << len(customers)) - 1)
Design an Ordered Stream

Simulation - just do what the problem says.

class OrderedStream:
    def __init__(self, n: int):
        self.a = [None] * n
        self.i = 0

    def insert(self, id: int, value: str) -> List[str]:
        id -= 1
        self.a[id] = value
        ans = []
        if self.i != id:
            return ans
        
        while self.i < len(self.a) and self.a[self.i] is not None:
            ans.append(self.a[self.i])
            self.i += 1
        
        return ans
Alex Wice — 11/15/2020
Determine if Two Strings Are Close

The problem essentially defines some equivalence classes.  One general method for knowing if two elements are equivalent is to create a canonical element of each equivalence class.

The first operation implies all anagrams are the same, so you only care about the frequencies.  You can swap frequencies so that the smallest character has the least frequency, and so on.

class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        def canonical(s):
            freqs = sorted(collections.Counter(s).values())
            ans = []
            keys = sorted(set(s))
            for i, f in enumerate(freqs):
                ans.extend(keys[i] * f)
            return ans
        
        return canonical(word1) == canonical(word2)
Alternate Solution

class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        count1 = collections.Counter(word1)
        count2 = collections.Counter(word2)
        return (
            sorted(count1.values()) == sorted(count2.values()) and
            sorted(count1) == sorted(count2)
        )
Minimum Operations to Reduce X to Zero

Instead of focusing on what we remove, let's focus on what we keep - we want the largest subarray with sum sum(A) - target.

This is now a standard sliding window problem - for each j, let's find the smallest i so that sum(A[i..j]) <= target.  The i's are monotone increasing.

class Solution:
    def minOperations(self, A: List[int], target: int) -> int:
        N = len(A)
        target = sum(A) - target

        ans = -1
        i = s = 0
        for j in range(N):
            s += A[j]
            while i < N and s > target:
                s -= A[i]
                i += 1
            if s == target:
                ans = max(ans, j - i + 1)
        
        if ans == -1:
            return ans
        return N - ans
Alternate Solution

For each possible number of things removed in the suffix (r_removed), let's see the number of l_removed to make the sum of the removed equal to the target.
class Solution:
    def minOperations(self, A, target):
        INF = float('inf')
        N = len(A)

        first = {0: 0}
        s = 0
        for i, x in enumerate(A, 1):
            s += x
            first.setdefault(s, i)
        
        ans = first.get(target, INF)
        for r_removed in range(1, N + 1):
            target -= A[-r_removed]
            l_removed = first.get(target, None)
            if l_removed is not None and l_removed + r_removed <= N:
                ans = min(ans, l_removed + r_removed)
        
        return ans if ans < INF else -1
Alex Wice — 12/06/2020
Maximize Grid Happiness

DP.  dp(r, c, ir, er, last) will store the max value of placing up to ir introverts, er extroverts, from (r, c) and on, where the last C placed people had value last.

class Solution:
    def getMaxGridHappiness(self, R, C, intro, extro):
        score = defaultdict(int)
        score[1, 1] = -60
        score[2, 1] = score[1, 2] = -10
        score[2, 2] = 40
        
        @cache
        def dp(r, c, i_rem, e_rem, last):
            if r == R:
                return 0
            if c == C:
                return dp(r + 1, 0, i_rem, e_rem, last)
            
            ans = dp(r, c+1, i_rem, e_rem, last[1:] + (0,))
            up, left = last[0], last[-1]
            if i_rem:
                gain = 120 + score[1, up] + (c > 0) * score[1, left]
                ans = max(ans, gain + dp(r, c+1, i_rem - 1, e_rem, last[1:] + (1,)))
            
            if e_rem:
                gain = 40 + score[2, up] + (c > 0) * score[2, left]
                ans = max(ans, gain + dp(r, c+1, i_rem, e_rem - 1, last[1:] + (2,)))

            return ans

        return dp(0, 0, intro, extro, (0,) * C)
Check If Two String Arrays are Equivalent

Just join the strings.

class Solution:
    def arrayStringsAreEqual(self, word1, word2):
        return "".join(word1) == "".join(word2)
Smallest String With A Given Numeric Value

Because we want the smallest, let's try to increment from least significant to most significant character.  We'll store the number of increments we've made, instead of the actual characters, and convert later.

At the ith character (from right to left), we can increment at most delta.

class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        ans = [1] * n
        k -= n
        
        for i in range(n - 1, -1, -1):
            delta = min(k, 25)
            ans[i] += delta
            k -= delta
        
        return "".join(chr(ord('a') - 1 + x) for x in ans)
Ways to Make a Fair Array

Sweepline.  We have some bags left and right.  Let's look at nums = [2,1,6,4] as an example.

Initially, right = {2,1,6,4} and left = {}.  As we move the line over to be on top of 2, we ditch 2 from right.  When we move the line from 2 to 1, we ditch 1 from right and we add 2 to left.

Now, left and right will actually be the sums of the bags, by parity.  For example, instead of right = [A[0], A[1], A[2], A[3]] we will have right = [A[0] + A[2], A[1] + A[3]].

class Solution:
    def waysToMakeFair(self, nums: List[int]) -> int:
        left = [0, 0]
        right = [0, 0]
        for i, x in enumerate(nums):
            right[i % 2] += x
        
        ans = 0
        
        for i in range(len(nums)):
            right[i % 2] -= nums[i]
            if i:
                left[~i % 2] += nums[i - 1]
            
            if left[0] + right[1] == left[1] + right[0]:
                ans += 1

        return ans
Minimum Initial Energy to Finish Tasks

Consider whether we should do task (a1, r1) or (a2, r2) first.  Say we have E energy.
Doing tasks 1 then 2 will have E >= r1 and E - a1 >= r2, while 2 then 1 will have E >= r2 and E - a2 >= r1.

Managing these equations to be independent in terms of the task, we have E >= (a1 - r1) + (r1 + r2) vs E >= (a2 - r2) + (r1 + r2).  Thus, for two adjacent tasks (a1, r1) and (a2, r2), we should do the one with the smaller a_i - r_i first.

We can use this fact to bubble sort the tasks by a_i - r_i.  Now we just have to check what is the minimum energy we need to do the tasks in this specific order, so we can use a greedy algorithm.

class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda t: t[0] - t[1])

        ans = energy = 0
        for actual, req in tasks:
            delta = max(0, req - energy)
            energy += delta
            ans += delta
            energy -= actual

        return ans
Maximum Repeating Substring

Greedy: try the largest possible answer, then the second largest, etc., until it works.

class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        ans = len(sequence) // len(word)
        while word * ans not in sequence:
            ans -= 1
        return ans
Merge In Between Linked Lists

We need pivot1, pivot2 as the a-1th and a+b+1th nodes.  Then, we just link them up in the obvious way.

class Solution:
    def mergeInBetween(self, list1, a, b, list2) -> ListNode:
        cur = list1
        for _ in range(a - 1):
            cur = cur.next
        
        pivot1 = cur
        for _ in range(b - a + 2):
            cur = cur.next
        
        pivot2 = cur
        
        cur = list2
        while cur.next:
            cur = cur.next
        
        pivot1.next = list2
        cur.next = pivot2
        return list1
Alex Wice — 12/06/2020
Design Front Middle Back Queue

Maintain the first and last half of the queue, where len(first) <= len(last) <= len(first) + 1.

class FrontMiddleBackQueue(object):
    def __init__(self):
        self.first = deque()
        self.last = deque()

    def pushFront(self, val):
        self.first.appendleft(val)
        self._rebalance()

    def pushMiddle(self, val):
        self.first.append(val)
        self._rebalance()

    def pushBack(self, val):
        self.last.append(val)
        self._rebalance()

    def popFront(self):
        if self.first:
            ans = self.first.popleft()
        elif self.last:
            ans = self.last.popleft()
        else:
            ans = -1
        self._rebalance()
        return ans

    def popMiddle(self):
        if len(self.first) < len(self.last):
            return self.last.popleft()
        elif self.first:
            return self.first.pop()
        else:
            return -1

    def popBack(self):
        if self.last:
            ans = self.last.pop()
        elif self.first:
            ans = self.first.pop()
        else:
            ans = -1
        self._rebalance()
        return ans

    def _rebalance(self):
        # |first| <= |last| <= |first| + 1
        if len(self.first) > len(self.last):
            self.last.appendleft(self.first.pop())
        if len(self.first) < len(self.last) - 1:
            self.first.append(self.last.popleft())
Minimum Number of Removals to Make Mountain Array

Say the final array has a peak at A[i].  Then the chosen elements until A[i] are increasing, and the elements chosen at and after A[i] are decreasing.  We can use two DPs to find the maximum lengths of these sequences.

Note that we need the check dp[i] >= 2 and ep[i] >= 2 to not fail the case [1,2,1,3,4,4].

We can also replace our O(N^2) dp with the O(N log N) dp for a better complexity.

class Solution:
    def minimumMountainRemovals(self, A: List[int]) -> int:
        N = len(A)
        
        # dp[i]: length of longest incr subsequence ending at A[i]
        dp = [1] * N
        for j in range(N):
            for i in range(j):
                if A[i] < A[j]:
                    dp[j] = max(dp[j], 1 + dp[i])
        
        # ep[i]: length of longest decr subsequence starting at A[i]
        ep = [1] * N
        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N):
                if A[i] > A[j]:
                    ep[i] = max(ep[i], 1 + ep[j])

        ans = N
        for i in range(1, N - 1):
            if dp[i] >= 2 and ep[i] >= 2:
                length = dp[i] + ep[i] - 1
                ans = min(ans, N - length)
        
        return ans
class Solution:
    def minimumMountainRemovals(self, A: List[int]) -> int:
        N = len(A)
        
        def LIS(A):
            ans = []
            dp = []
            for x in A:
                i = bisect.bisect_left(dp, x)
                if i >= len(dp):
                    dp.append(0)
                dp[i] = x
                ans.append(i + 1)
            return ans
        
        dp = LIS(A)
        ep = LIS(A[::-1])[::-1]

        ans = N
        for i in range(1, N - 1):
            if dp[i] >= 2 and ep[i] >= 2:
                length = dp[i] + ep[i] - 1
                ans = min(ans, N - length)
        
        return ans
Richest Customer Wealth

Each customer's wealth is the sum of that customer's row.

class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        return max(map(sum, accounts))
Find the Most Competitive Subsequence

Let's maintain the correct answer.  Say we have [2, 3, 4, 5, 1, ...].  What should happen when we see 1?  We delete as many elements as possible, ideally starting our answer with [1, ...].

We can use a stack to keep track of the answer, and also keep track of how many elements we have to remove.

class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        to_remove = len(nums) - k
        stack = []

        for x in nums:
            while stack and x < stack[-1] and to_remove:
                to_remove -= 1
                stack.pop()
            stack.append(x)
        
        for _ in range(to_remove):
            stack.pop()
        
        return stack
Alternate Solution

Let's build the answer from left to right.  We start with the first n - k + 1 elements, choosing the smallest to start our answer with.  Then, we add the next element and choose the next smallest (that occurs to the right of what was previously chosen), and so on.

We use a heap to maintain everything.

from heapq import *

class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        pq = [[nums[i], i] for i in range(n - k)]
        heapify(pq)
        
        ans = []
        last = -1
        for i in range(n - k, n):
            heappush(pq, [nums[i], i])
            while pq[0][1] < last:
                heappop(pq)
            
            val, index = heappop(pq)
            last = max(last, index)
            ans.append(val)
            
        return ans
Minimum Moves to Make Array Complementary

Let's say the final array has all pairs summing to s.
The pair nums[i], nums[n - 1 - i] = [x, y] (with say, x <= y) will take:

* 0 moves if the sum is already s, else
* 1 move if the sum is in [x + 1, y + limit], else
* 2 moves.

To add these intervals efficiently, we can use suffix sums.  add(lo, hi, v) will add v to the range [lo..hi], by a difference of two suffix sums: A[lo] += v adds v to every index >= lo, and A[hi + 1] subtracts v from every index >= hi + 1.

At the end, we convert these suffix events to the actual array before taking the minimum.

class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        A = [0] * (2 * limit + 1)
        
        def add(lo, hi, v):
            A[lo] += v
            if hi < 2 * limit:
                A[hi + 1] -= v

        for i in range(n >> 1):
            x, y = nums[i], nums[~i]
            if x > y:
                x, y = y, x
            
            add(0, 2 * limit, 2)
            add(x + 1, y + limit, -1)
            add(x + y, x + y, -1)

        for i in range(1, len(A)):
            A[i] += A[i - 1]
        
        return min(events)
Minimize Deviation in Array

Let's say you can only decrease numbers (use operation type-1).  The answer is the minimum deviation while we greedily decrease the largest number.

But actually, the answer for an array with odd numbers is the same as if the array had all odd numbers multiplied by two.

We can use a maxheap (min heap of negative elements) to get the next maximum number to decrease, while keeping track of low, the lowest number.

class Solution:
    def minimumDeviation(self, A) -> int:
        pq = [-x * 2 if x & 1 else -x for x in A]
        heapq.heapify(pq)
        
        ans = float('inf')
        low = -max(pq)
        while True:
            top = -heapq.heappop(pq)
            ans = min(ans, top - low)
            if top & 1:
                break
            top >>= 1
            low = min(low, top)
            heapq.heappush(pq, -top)
        
        return ans
Goal Parser Interpretation

Just replace.

class Solution:
    def interpret(self, s: str) -> str:
        return s.replace('()', 'o').replace('(al)', 'al')


Alternate Solution

class Solution:
    def interpret(self, s: str) -> str:
        ans = []
        i = 0
        while i < len(s):
            if s[i] == 'G':
                ans.append('G')
                i += 1
            elif s[i + 1] == ')':  # ()
                ans.append('o')
                i += 2
            else:  # (al)
                ans.append('al')
                i += 4
        
        return "".join(ans)
Max Number of K-Sum Pairs

Keep track of all the unpaired elements we've seen in a multiset count.

When we find a matching element, pair them and increase the answer, otherwise add them to the count to await being paired.

class Solution:
    def maxOperations(self, A: List[int], target: int) -> int:
        count = Counter()
        ans = 0
        for x in A:
            if count[target - x]:
                count[target - x] -= 1
                ans += 1
            else:
                count[x] += 1
        
        return ans
Concatenation of Consecutive Binary Numbers

Precompute all the answers.

If we have say, the answer for 4 (prev = 0b11011100), the next value is nxt = prev << 3 + 5.

class Solution:
    MOD = 10 ** 9 + 7
    A = [0]
    for x in range(1, 10 ** 5 + 1):
        A.append(((A[-1] << x.bit_length()) + x) % MOD)
    
    def concatenatedBinary(self, n: int) -> int:
        return Solution.A[n]
Alex Wice — 12/06/2020
Minimum Incompatibility

DP.  The problem has a nice substructure because after we create one bucket, the remaining elements have a sum of amplitudes ("incompatibilities").

Let dp(values) be the remaining values to distribute into buckets.  We can choose some and check manually that there are no repeats.

Alternatively, we can do a mask dp.

class Solution:
    def minimumIncompatibility(self, A: List[int], num_buckets: int) -> int:
        INF = float('inf')
        B = len(A) // num_buckets  # size of each bucket

        @lru_cache(None)
        def dp(remaining):
            # remaining: values of elements that remain to be chosen
            if not remaining:
                return 0
            
            ans = INF
            for choice in combinations(remaining, B):
                if len(set(choice)) == B:
                    remaining2 = list(remaining)
                    for c in choice:
                        remaining2.remove(c)
                    ans = min(ans, max(choice) - min(choice) + dp(tuple(remaining2)))
            
            return ans
        
        ans = dp(tuple(A))
        return ans if ans < INF else -1
Alternative Solution
class Solution {
public:
    int minimumIncompatibility(vector<int>& A, int num_buckets) {
        int N = A.size();
        int B = N / num_buckets;  // size of each bucket
        sort(A.begin(), A.end());

        map<int, int> amplitudes;  // valid mask -> amplitude of that mask
        vector<int> choice;

        for (int mask = 0; mask < (1 << N); ++mask)
            if (__builtin_popcount(mask) == B) {
                choice.clear();
                for (int i = 0; i < N; ++i) if (mask >> i & 1)
                    choice.emplace_back(A[i]);

                sort(choice.begin(), choice.end());
                choice.erase(unique(choice.begin(), choice.end()), choice.end());
                
                if (choice.size() == B)
                    amplitudes[mask] = choice[B - 1] - choice[0];
            }

        int INF = 1e9;
        vector<int> dp(1 << N, INF);
        dp[0] = 0;

        for (int mask = 1; mask < (1 << N); ++mask) if (__builtin_popcount(mask) % B == 0) {
            for (const auto& [s, v] : amplitudes) if (mask & s == s) {
                dp[mask] = min(dp[mask], dp[mask ^ s] + v);
            }
        }
        
        int ans = dp[(1 << N) - 1];
        return ans < INF ? ans : -1;
    }
};
Alex Wice — 12/28/2020
Count the Number of Consistent Strings

Let's check to see if each word is consistent.  To check whether a character is allowed, we use a set to check in O(1) time.

class Solution:
    def countConsistentStrings(self, allowed, words):
        allowset = set(allowed)
        
        ans = 0
        for word in words:
            if all(c in allowset for c in word):
                ans += 1
        
        return ans
Sum of Absolute Differences in a Sorted Array

We can write the sum of the differences without the absolute value signs.

For example, if A = [2, 3, 5, 8], then ans[1] = [(3 - 2)] + [(5 - 3) + (8 - 3)].

This is made up of a "left" component of i * A[i] - sum(A[:i]), and a right component of sum(A[i+1:]) - (N - 1 - i) * A[i].

We can keep track of lsum = sum(A[:i]) and rsum = sum(A[i+1:]) to help us write each ans[i].

class Solution:
    def getSumAbsoluteDifferences(self, A: List[int]) -> List[int]:
        N = len(A)
        lsum = 0
        rsum = sum(A)
        ans = [0] * N
        
        for i in range(N):
            rsum -= A[i]

            ans[i] = rsum - A[i] * (N - 1 - i) + i * A[i] - lsum
            
            lsum += A[i]
        
        return ans
Stone Game VI

Say the score of the game is (Alice's points) - (Bob's points).  The game is equivalent to Alice trying to make the score positive and Bob trying to make the score negative.

Alice taking pile i increases the score by A[i] + B[i] (and Bob taking it decreases the score also by A[i] + B[i]).

Therefore, at each turn each player just takes the most important pile (the one with the largest A[i] + B[i].)

class Solution:
    def stoneGameVI(self, A: List[int], B: List[int]) -> int:
        piles = sorted(zip(A, B), key=sum, reverse=True)
        alice = sum(a for a, b in piles[::2])
        bob = sum(b for a, b in piles[1::2])
        
        return 1 if alice > bob else -1 if alice < bob else 0
Delivering Boxes from Storage to Ports

Basically, the ship always greedily takes the most boxes it can each trip.  The only exception is that it may return home before visiting the last port in the greedy trip.  This is because having to visit the same port twice on two separate trips can add 1 to the answer.

Let dp[i] be the answer for delivering boxes A[:i].  Say the greedy trip goes to A[:j].  There are two cases, we either end the trip at the end of the last port change (anchor), or at j.

These anchor, j's are monotone increasing, so we use a sliding window.

class Solution:
    def boxDelivering(self, A, num_ports, max_boxes, max_weight):
        N = len(A)
        dp = [float('inf')] * (N + 1)
        dp[0] = 0

        box_cap = max_boxes
        wei_cap = max_weight
        j = 0
        port_changes = 0
        for i in range(N):
            
            while j < N and box_cap and wei_cap >= A[j][1]:
                box_cap -= 1
                wei_cap -= A[j][1]
                
                if j == 0 or A[j][0] != A[j-1][0]:
                    anchor = j
                    port_changes += 1

                j += 1

            dp[j] = min(dp[j], dp[i] + port_changes + 1)
            dp[anchor] = min(dp[anchor], dp[i] + port_changes)
            
            box_cap += 1
            wei_cap += A[i][1]
            if i + 1 < N and A[i][0] != A[i+1][0]:
                port_changes -= 1
            
        return dp[-1]
Count of Matches in Tournament

There are n - 1 losers in the tournament, and each loser played exactly one match.

class Solution:
    def numberOfMatches(self, n):
        return n - 1
Partitioning Into Minimum Number Of Deci-Binary Numbers

As we never "carry", we need atleast ans = int(max(s)) numbers.  Also, we have a construction for the answer ans: the k-th number is all zeroes, except a '1' for every digit <= k.  (Eg. "32" = "11" + "11" + "10").

class Solution:
    def minPartitions(self, s):
        return int(max(s))
Alex Wice — 12/28/2020
Stone Game VII

Let the "score" refer to the difference in Alice's and Bob's scores.

Let dp[i][j] be the score of the game for stones[i..j].

The transitions are you either take the ith stone or the jth stone, and the score is eg. sum(A[i+1..j]) - dp[i+1][j] for taking the ith stone.  We use prefix sums (P[j] = sum(A[:j])) to calculate sum(A[..]) quickly.

class Solution:
    def stoneGameVII(self, A):
        N = len(A)
        dp = [[0] * N for _ in range(N)]
        P = [0]
        for x in A:
            P.append(P[-1] + x)

        for i in range(N - 2, -1, -1):
            for j in range(i + 1, N):
                dp[i][j] = max(P[j + 1] - P[i + 1] - dp[i + 1][j], P[j] - P[i] - dp[i][j - 1]);
        return dp[0][N - 1]
Maximum Height by Stacking Cuboids

Due to the unusual condition of the problem, we can always have the cuboids rotated in sorted order of dimension.

Then, we can stack A[j] ontop of A[i] only when sorted(A[i]) <= sorted(A[j]).

After, this becomes a "longest increasing subsequence" problem, which is easy to solve in O(N^2) (or better.)

class Solution:
    def maxHeight(self, A: List[List[int]]) -> int:
        N = len(A)
        A = sorted(map(sorted, A))  

        # dp[i] is the max height for stacking boxes ending in A[i]
        dp = [z for x, y, z in A]
        
        for j in range(N):
            for i in range(j):
                if all(A[i][k] <= A[j][k] for k in range(3)):
                    dp[j] = max(dp[j], dp[i] + A[j][2])
        
        return max(dp)
Reformat Phone Number

Use str.replace to remove the spaces and dashes.

Now group them in threes.  This almost works, except when there is a string of length 1 remaining, which you can fix.

class Solution:
    def reformatNumber(self, s):
        s = s.replace(' ', '').replace('-', '')
        ans = []
        for i in range(0, len(s), 3):
            ans.append(s[i:i+3])
        if ans and len(ans[-1]) == 1:
            ans[-1] = ans[-2][-1] + ans[-1]
            ans[-2] = ans[-2][:2]
        
        return "-".join(ans)
Maximum Erasure Value

Sliding window.  For each j, let i = opt(j) be the smallest i so that A[i..j] has distinct elements.

opt(j) is monotone increasing, so we can use a sliding window.

After, A[i..j] is the best candidate erasure that ends at A[j], because each value in the array is positive.

class Solution:
    def maximumUniqueSubarray(self, A):
        P = [0]
        for x in A:
            P.append(P[-1] + x)
        
        count = Counter()
        ans = i = 0
        for j in range(len(A)):
            # Find the smallest i so that A[i..j] has distinct elements.
            count[A[j]] += 1
            while count[A[j]] == 2:
                count[A[i]] -= 1
                i += 1
            
            ans = max(ans, P[j + 1] - P[i])
        
        return ans
Jump Game VI

The "obvious" DP is: let dp[i] be the maximum score to jump to i.

Then dp[i] = A[i] + max(dp[i-1], dp[i-2], ..., dp[i-K]).

To get max(...) efficiently, we can use a MaxQueue.

Alternatively, we show another solution using a priority queue to handle the max operation.

class MaxQueue(collections.deque):
    """
    A queue with O(1) max operations.
    """
    def enqueue(self, val):
        count = 1
        while self and self[-1][0] < val:
            count += self.pop()[1]
        self.append([val, count])

    def dequeue(self):
        ans = self.max()
        self[0][1] -= 1
        if self[0][1] <= 0:
            self.popleft()
        return ans

    def max(self):
        return self[0][0] if self else 0


class Solution(object):
    def maxResult(self, A, K):
        mq = MaxQueue()
        for i, x in enumerate(A):
            ans = x + mq.max()
            mq.enqueue(ans)
            if i >= K:
                mq.dequeue()
        return ans
class Solution(object):
    def maxResult(self, A, K):
        pq = []
        todel = Counter()
        dp = []

        for i, x in enumerate(A):
            while pq and todel[pq[0]] > 0:
                todel[heappop(pq)] -= 1
            ans = x + (-pq[0] if pq else 0)
            dp.append(ans)
            heappush(pq, -ans)
            if i >= K:
                todel[-dp[i-K]] += 1

        return dp[-1]
Checking Existence of Edge Length Limited Paths

The queries (u, v, limit) concern whether u and v are connected in some graph G_limit (the given graph with only edges of weight <= limit), so DSU (Disjoint Set Union) seems like an obvious choice (since it helps you respond to connectivity queries.)

The graphs G_limit are subgraphs of each other if we take limit in sorted order.  So we can use just one graph and add edges as necessary to bring the previous graph up to be equal to G_limit.

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]

class Solution:
    def distanceLimitedPathsExist(self, n, edges, queries):
        dsu = DSU(n)
        ans = [False] * len(queries)
        for qid, query in enumerate(queries):
            query.append(qid)
        
        edges.sort(key=lambda edge: edge[2])
        queries.sort(key=lambda query: query[2])
        
        i = 0
        for p, q, limit, qid in queries:
            while i < len(edges) and edges[i][2] < limit:
                dsu.union(edges[i][0], edges[i][1])
                i += 1
            
            if dsu.find(p) == dsu.find(q):
                ans[qid] = True
        
        return ans
Number of Students Unable to Eat Lunch

The student queue is a red herring.  The only thing that matters is if there is a student that is willing to eat the next sandwich - the line will rotate so that they eat it.

Let count[x] be the number of students that want to eat sandwiches of type x.  We maintain this count as we go through the requirements.

class Solution:
    def countStudents(self, students, sandwiches):
        count = [students.count(0), students.count(1)]
        
        for req in sandwiches:
            if count[req] == 0:
                break
            else:
                count[req] -= 1
        
        return sum(count)
Average Waiting Time

Say a customer arrives at arrival and it takes length time to prepare their order.

The time when the order will be done is max(prev_time, arrival) + length.

The total time that customer was waiting will be time - arrival.

The average time is total_time / len(customers).

class Solution:
    def averageWaitingTime(self, customers):
        ans = time = 0
        for arrival, length in customers:
            time = max(time, arrival) + length
            ans += time - arrival
        
        return ans / len(customers)
Maximum Binary String After Change

Because we want the lexicographically largest string, getting "1"'s in the most significant position dominates all other objectives.

We can always ignore any "1"s in the prefix of the string: say there are prefix of them.

When you have a "0" in the prefix, the next "0" found can be used by the "10" -> "01" transformation to "pull" the string into eg. "011110 -> ""001111" and then to -> "101111".  The net result of this is that the second zero is erased and prefix += 1.

This means that the "states" are only empty, or "01*" ("0" with some number of "1"s after [possibly zero]).

We now go left to right and update the transitions in our state machine as described.

class Solution:
    def maximumBinaryString(self, binary: str) -> str:
        state = 0  # 0: empty, X: "0" + "1" * (X - 1)
        prefix = 0
        
        for d in map(int, binary):
            if state == 0:
                if d == 0:
                    state += 1
                else:
                    prefix += 1
            else:
                if d == 0:
                    prefix += 1
                else:
                    state += 1

        if state == 0:
            return '1' * prefix 
        else:
            return '1' * prefix + '0' + '1' * (state - 1)
Minimum Adjacent Swaps for K Consecutive Ones

Say the ones are at positions inds[0], inds[1], ....  Let's try to make inds[i], ..., inds[i + K - 1] consecutive.

Evidently, this means minimizing f(c) = |inds[i] - c| + |inds[i+1] - (c+1)| + ... + |inds[i+K-1] - (c+K-1)| over all choices c.

Let jnds[i] = inds[i] - i.  Then f(c) = |jnds[i] - c| + |jnds[i+1] - c| + ... + |jnds[i+K-1] - c|.

This function is minimized when c = median(jnds[i:i+K]).  (This is because when c is not the median, we can make it smaller by pushing it towards the median as more |...| terms will decrease than increase.)

Now jnds[m] is the median, and we can directly calculate the absolute value sum (its the "left half" [the stuff <= median], plus the right half.)

class Solution:
    def minMoves(self, A, K):
        N = len(A)
        inds = [i for i, x in enumerate(A) if x]
        jnds = [x-i for i, x in enumerate(inds)]
        P = list(itertools.accumulate(jnds, initial=0))
            
        ans = inf
        for i in range(len(jnds) - K + 1):
            j = i + K - 1
            m = i + j >> 1
            # jnds[i..j] ones will be put together
            # median position is jnds[m]
            
            cand = (m - i + 1) * jnds[m] - (P[m + 1] - P[i])
            cand += (P[j + 1] - P[m]) - (j - m + 1) * jnds[m]
            ans = min(ans, cand)
        
        return ans
Determine if String Halves Are Alike

Iterate over each vowel s[i].  If i is in the first half, bal += 1, else bal -= 1.  We want bal (the balance) to be zero.

class Solution(object):
    def halvesAreAlike(self, s):
        bal = 0
        vowels = set('aeiouAEIOU')
        for i, c in enumerate(s):
            if c in vowels:
                if i < len(s) // 2:
                    bal += 1
                else:
                    bal -= 1
        return bal == 0
Maximum Number of Eaten Apples

Let's simulate time going forward.  Evidently, with a collection of apples, you always want to eat the apples that expire first.  Since this is a min query, we use a heap that keeps track of the expiry and quantity of each batch of apples.

Now let's go to the algorithm.  At the start of a day, add apples to our collection if necessary.  Then, discard all batches of apples that are expired.  Finally, if there is a batch of apples left over, eat one and ans += 1, plus discard the batch if its now empty.

class Solution:
    def eatenApples(self, apples, days):
        n = len(apples)
        
        ans = 0
        pq = []  # heap of [expiry, qty]
        for t in range(4 * 10 ** 4):
            if t < n:
                expiry = t + days[t] - 1
                heappush(pq, [expiry, apples[t]])
            
            while pq and pq[0][0] < t:
                heappop(pq)

            if pq:
                ans += 1
                pq[0][1] -= 1
                if pq[0][1] == 0:
                    heappop(pq)
        
        return ans
Where Will the Ball Fall

Let roots[c] = [indices of balls at column c].  We should update roots as we process each row of the grid.

Evidently, only grid values of \\ ([1,1]) or // ([-1,-1]) will have the ball move to the next row.  We can handle these transitions appropriately.

class Solution:
    def findBall(self, grid):
        C = len(grid[0])
        
        # roots[c] = the ball indices that are at this column
        roots = [[c] for c in range(C)]
        for row in grid:
            nroots = [[] for _ in range(C)]
            for c in range(C - 1):
                if row[c] == row[c + 1] == 1:
                    nroots[c + 1].extend(roots[c])
                if row[c] == row[c + 1] == -1:
                    nroots[c].extend(roots[c + 1])
            
            roots = nroots
        
        ans = [-1] * C
        for c, col in enumerate(roots):
            for i in col:
                ans[i] = c
        
        return ans
Alex Wice — 12/28/2020
**Maximum XOR With an Element From Array

Consider first the following smaller problem: given A: List[int] and x, return max(x ^ y for y in A).

The answer is to put the values of A (in binary) into a trie.  Then, at each bit (from most to least significant) we want to take the path in the trie corresponding to the largest bit possible.

Now if we sort the queries (x, m), we can insert only the values of A into the trie that are <= m as we go.
class TrieNode:
    def __init__(self):
        self.children = [None, None]
    
    def __getitem__(self, v):
        if self.children[v] is None:
            self.children[v] = TrieNode()
        return self.children[v]
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, x):
        # insert x into the trie
        cur = self.root
        for i in range(30, -1, -1):
            bit = x >> i & 1
            cur = cur[bit]

    def query(self, x):
        # return max(x ^ y for y in trie)
        cur = self.root
        ans = 0
        for i in range(30, -1, -1):
            want = (x >> i & 1) ^ 1
            if cur.children[want] is not None:
                cur = cur[want]
                ans = 2 * ans + want
            elif cur.children[want ^ 1] is not None:
                cur = cur[want ^ 1]
                ans = 2 * ans + want ^ 1
            else:
                return -1
        
        return ans ^ x

class Solution:
    def maximizeXor(self, A, queries):
        A.sort()
        trie = Trie()
        
        for qid, query in enumerate(queries):
            query.append(qid)
        queries.sort(key=lambda q: q[1])

        ans = [-1] * len(queries)
        i = 0
        for x, m, qid in queries:
            # Add all elements <= m from A to the trie.
            while i < len(A) and A[i] <= m:
                trie.insert(A[i])
                i += 1
            ans[qid] = trie.query(x)

        return ans
We can also solve the problem "online".  For each TrieNode node, let node.min be the minimum value inserted that passes through the subtree node.  Then, we know if a greedy choice down our trie when querying is valid, by checking node.min.
# This code TLEs on LeetCode, but it is more instructive for interviews than hacking for performance.

class TrieNode:
    def __init__(self):
        self.children = [None, None]
        self.min = float('inf')
    
    def __getitem__(self, v):
        if self.children[v] is None:
            self.children[v] = TrieNode()
        return self.children[v]
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, x):
        # insert x into the trie
        cur = self.root
        cur.min = min(cur.min, x)
        for i in range(30, -1, -1):
            bit = x >> i & 1
            cur = cur[bit]
            cur.min = min(cur.min, x)

    def query(self, x, m):
        # return max(x ^ y for y in trie if y <= m)

        cur = self.root
        ans = 0
        for i in range(30, -1, -1):
            want = (x >> i & 1) ^ 1
            if cur.children[want] is not None and cur[want].min <= m:
                cur = cur[want]
                ans = 2 * ans + want
            elif cur.children[want ^ 1] is not None and cur[want ^ 1].min <= m:
                cur = cur[want ^ 1]
                ans = 2 * ans + want ^ 1
            else:
                return -1
        
        return ans ^ x

class Solution:
    def maximizeXor(self, A, queries):
        trie = Trie()
        for x in A:
            trie.insert(x)

        ans = [-1] * len(queries)
        for qid, (x, m) in enumerate(queries):
            ans[qid] = trie.query(x, m)
        
        return ans
Alex Wice — 01/17/2021
Maximum Units on a Truck

Greedy.  We always want to put the boxes with the most units onto the truck first.

class Solution(object):
    def maximumUnits(self, shipments, truck_capacity):
        shipments.sort(key=lambda e: -e[1])
        ans = 0
        for nboxes, units in shipments:
            delta = min(nboxes, truck_capacity)
            truck_capacity -= delta
            ans += delta * units
        
        return ans
Count Good Meals

Let's count good meals that sum to 2 ** 0, 2 ** 1, 2 ** 2, ... separately.

For each such power pwr = 2 ** expo, we can count it using a kind of brute force: for each meal x, we need to know how many previous meals we've seen with value pwr - x.  We can use a collections.Counter.

class Solution:
    def countPairs(self, A):
        ans = 0
        for expo in range(22):
            pwr = 1 << expo
            
            # How many pairs in A sum to pwr?
            count = Counter()
            for x in A:
                ans += count[pwr - x]
                count[x] += 1
        
        return ans % (10 ** 9 + 7)
Ways to Split Array Into Three Subarrays

Let P[i] = A[0] + ... + A[i-1], the i-th prefix sum.

We can phrase the condition left <= mid <= right in terms of prefix sums:

left = P[i] - P[0], mid = P[j] - P[i], right = P[-1] - P[j]

Now for each such i, the interval [j1, j2) that represents all the valid j is monotone increasing in j1 and j2, so we can use a sliding window.

class Solution:
    def waysToSplit(self, A):
        MOD = 10 ** 9 + 7
        P = list(accumulate(A, initial=0))
        
        ans = 0
        j1 = j2 = 0
        
        # Pi <= Pj - Pi <= S - Pj
        # 2Pi <= Pj  and 2Pj <= S + Pi
        for i in range(1, len(P)):
            j1 = max(j1, i + 1)
            while j1 < len(P) and 2 * P[i] > P[j1]:
                j1 += 1

            j2 = max(j2, j1)
            while j2 < len(P) - 1 and 2 * P[j2] <= P[-1] + P[i]:
                j2 += 1

            ans += j2 - j1
        
        return ans % MOD
Minimum Operations to Make a Subsequence

DP similar to Longest Increasing Subsequence (N log N solution).  If you are not familiar with it, try that problem first.

Let dp[steps] = i be the minimum index i such that A[..i] needs steps moves so that T[] is a subsequence of A[..i].

Now dp is monotone increasing, so we can binary search on it to maintain it throughout the iteration.

class Solution:
    def minOperations(self, T: List[int], A: List[int]) -> int:
        locs = collections.defaultdict(list)
        for i, x in enumerate(A):
            locs[x].append(i)
        
        # Let dp[steps] = i,
        #    the minimum index i such that A[..i] needs steps
        #    moves so that T[] is a subsequence of A[..i]
        dp = [-1]  # increasing sequence
        for t in T:
            for i in reversed(locs[t]):
                ix = bisect.bisect_left(dp, i)
                if ix >= len(dp):
                    dp.append(-1)
                dp[ix] = i

        return len(T) + 1 - len(dp)
Calculate Money in Leetcode Bank

Simulation.  Keep track of the week and day's number.

class Solution:
    def totalMoney(self, n: int) -> int:
        week = day = 0
        for _ in range(n):
            ans += week + day
            day += 1
            if day == 8:
                day = 1
                week += 1
        return ans


Bonus O(1) solution

class Solution:
    def totalMoney(self, n: int) -> int:
        q, r = divmod(n - 1, 7)
        # q : total number of completed weeks
        # r : day (0 indexed) of the last week that wasnt completed

        ans = 28 * q + q * (q - 1) * 7 // 2
        ans += (r + 1) * (r + 2) // 2
        ans += q * (r + 1)
        return ans
Maximum Score From Removing Substrings

Look at groups of ('a' or 'b') characters - the answer is a sum of these independent groups.

Now there are two cases: ab_score >= ba_score or the opposite.  In either case, we use a greedy solution to take all "ab"'s first, then all remaining "ba"'s.

class Solution:
    def maximumGain(self, S, ab_score, ba_score):
        ans = 0
        for key, grp in groupby(S, key=lambda c: c in 'ab'):
            if key:
                if ab_score >= ba_score:
                    a = b = 0
                    for c in grp:
                        if c == 'b':
                            if a:
                                a -= 1
                                ans += ab_score
                            else:
                                b += 1
                        else:
                            a += 1
                    
                    ans += min(a, b) * ba_score
                else:
                    a = b = 0
                    for c in grp:
                        if c == 'a':
                            if b:
                                b -= 1
                                ans += ba_score
                            else:
                                a += 1
                        else:
                            b += 1
                    
                    ans += min(a, b) * ab_score
        
        return ans
Construct the Lexicographically Largest Valid Sequence

Backtracking.  seq is the current sequence we are writing, and used[x] is whether the number x has been used.

class Solution:
    def constructDistancedSequence(self, n: int) -> List[int]:
        seq = [0] * (2 * n - 1)
        used = [False] * (n + 1)

        def search(i):
            if i == len(seq):
                return True
            if seq[i]:
                return search(i + 1)
            
            for x in range(n, 1, -1):
                if used[x]:
                    continue
                
                if i + x < len(seq) and seq[i + x] == 0:
                    seq[i] = seq[i + x] = x
                    used[x] = True
                    if search(i + 1):
                        return True
                    used[x] = False
                    seq[i] = seq[i + x] = 0
            
            if not used[1]:
                seq[i] = 1
                used[1] = True
                if search(i + 1):
                    return True
                used[1] = False
                seq[i] = 0
        
        search(0)
        return seq        
Number Of Ways To Reconstruct A Tree

The root of the tree has to be connected to every other node.  We can remove the edges from root to every other node, which creates some connected components that we can recursively solve.

class Solution:
    def checkWays(self, pairs: List[List[int]]) -> int:
        graph = defaultdict(set)
        for u, v in pairs:
            graph[u].add(v)
            graph[v].add(u)
        
        def solve(nodes):
            root_count = 0
            for node in nodes:
                if len(graph[node]) == len(nodes) - 1:
                    root = node
                    root_count += 1
            
            ans = 1
            if root_count == 0:
                return 0
            elif root_count > 1:
                ans = 2
            
            # root is the desired root
            for nei in graph[root]:
                graph[nei].discard(root)
            
            seen = set()
            for start in graph[root]:
                if start in seen:
                    continue
                queue = [start]
                seen.add(start)
                for node in queue:
                    for nei in graph[node] - seen:
                        queue.append(nei)
                        seen.add(nei)
                
                # Now queue is the connected component
                bns = solve(queue)
                if bns == 0:
                    ans = 0
                if ans and bns == 2:
                    ans = 2
            
            return ans
        
        return solve(list(graph))
Decode XORed Array

XORing by A[i] both sides of the equation encoded[i] = A[i] XOR A[i+1], we have A[i+1] = encoded[i] XOR A[i].  This is a simple recursion for A.

class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        ans = [first]
        for x in encoded:
            ans.append(ans[-1] ^ x)
        return ans
Swapping Nodes in a Linked List

Put slow = head and fast = head[k].  Then moving fast to the end, slow will be at head[~k]`.

Now we can swap the values.

class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        slow = fast = head
        for _ in range(k - 1):
            fast = fast.next
        
        first = fast
        while fast.next:
            slow = slow.next
            fast = fast.next
        
        second = slow
        
        first.val, second.val = second.val, first.val
        return head
Minimize Hamming Distance After Swap Operations

Consider the allowed swaps as edges in a graph.  Being able to swap from the edges at will is equivalent to being able to permute the connected components of this graph in any way we want.

Now for each component, we can count the source letters and target (sink) letters.  The "unsinked" letters in the source are the ones we need to replace.

class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        return True

    def size(self, x):
        return self.sz[self.find(x)]

class Solution:
    def minimumHammingDistance(self, source, target, edges):
        n = len(source)
        dsu = DSU(n)
        for u, v in edges:
            dsu.union(u, v)
        
        components = defaultdict(list)
        for u in range(n):
            components[dsu.find(u)].append(u)
        
        ans = 0
        for component in components.values():
            count = defaultdict(int)
            for i in component:
                count[source[i]] += 1
                count[target[i]] -= 1
            ans += sum(v for v in count.values() if v > 0)
        
        return ans
Find Minimum Time to Finish All Jobs

Binary search for the answer.  We can test whether some bound is possible with a backtracking algorithm.

class Solution:
    def minimumTimeRequired(self, jobs: List[int], K: int) -> int:
        n = len(jobs)
        jobs.sort(reverse=True)
        
        def possible(bound):
            work = [0] * K
            def search(i):
                if i == len(jobs):
                    if max(work) <= bound:
                        return True
                
                for k in range(K):
                    if work[k] + jobs[i] <= bound and (k == 0 or work[k] != work[k-1]):
                        work[k] += jobs[i]
                        if search(i + 1):
                            return True
                        work[k] -= jobs[i]
                
                return False

            return search(0)

        lo, hi = 0, sum(jobs)
        while lo < hi:
            mi = lo + hi >> 1
            if not possible(mi):
                lo = mi + 1
            else:
                hi = mi
        
        return lo
Number Of Rectangles That Can Form The Largest Square

Do what it says in the problem: find the square lengths, find the largest length m, and find the number of times m occurs.

class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        lengths = list(map(min, rectangles))
        m = max(lengths)
        return lengths.count(m)
Tuple with Same Product

Count the occurrences of A[i] * A[j] (with i < j).

Now let's say A[i1] * A[j1] = A[i2] * A[j2] = p.  Using math, we can show that i1, j1, i2, j2 must be different indices.

If there are v = count[p] occurrences of some product, (say (i1, j1), ..., (iv, jv)), then there must be binom(v, 2) = v * (v - 1) // 2 choices for two tuples.  After, there are 8 ways to arrange these tuples.

class Solution:
    def tupleSameProduct(self, A):
        count = Counter()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                count[A[i] * A[j]] += 1
        
        ans = 0
        for v in count.values():
            ans += v * (v - 1) // 2
        
        return ans * 8
Largest Submatrix With Rearrangements

Let A[r][c] be the most number of 1's you can find going up from (r, c) until you hit a 0.

The motivation for this is that from each row, you can know what the largest rectangle is, for example if you see [5,2,3,6] then the largest height you can make is 2, so the total rectangle area is 4 * 2.

Now sort each row (we can because we are allowed to sort columns).  For each row[c], it is the minimum height for row[c:] which has a width of C - c.

class Solution:
    def largestSubmatrix(self, A):
        R, C = len(A), len(A[0])
        for c in range(C):
            s = 0
            for r in range(R):
                s = 0 if A[r][c] == 0 else s + 1
                A[r][c] = s
        
        ans = 0
        for srow in map(sorted, A):
            for c in range(C):
                ans = max(ans, (C - c) * srow[c])
        
        return ans
Cat and Mouse II

DP.  decide(mr, mc, cr, cc, turn) returns either MOUSE or CAT depending on who wins (with the mouse at (mr, mc) ["mouse row", "mouse column"], cat at (cr, cc), and it is the turn-th turn.)

If the mouse or cat is at the food, or the cat is at the mouse, or it has been too many turns, the game is over.

Otherwise, we attempt for the mouse (or cat, if it's the cat's turn), to jump in one of the four directions and return if they can still win the game.

There are no cycles because of the turn part of the DP, so everything works out.
class Solution:
    def canMouseWin(self, A, cat_jump, mouse_jump):
        MOUSE, CAT = 0, 1
        R, C = len(A), len(A[0])
        for r in range(R):
            for c in range(C):
                if A[r][c] == 'M':
                    MR, MC = r, c
                elif A[r][c] == 'C':
                    CR, CC = r, c
                elif A[r][c] == 'F':
                    FR, FC = r, c

        @functools.cache
        def decide(mr, mc, cr, cc, turn):
            if mr == FR and mc == FC:
                return MOUSE
            if mr == cr and mc == cc:
                return CAT
            if cr == FR and cc == FC:
                return CAT
            if turn >= 80:
                return CAT
            
            # enumerate neighbors
            if turn & 1 == 0:  # mouse's turn
                for dr, dc in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                    for k in range(mouse_jump + 1):
                        nr = mr + dr * k
                        nc = mc + dc * k
                        if not (0 <= nr < R and 0 <= nc < C):
                            break
                        if A[nr][nc] == '#':
                            break
                        if decide(nr, nc, cr, cc, turn + 1) == MOUSE:
                            return MOUSE
                return CAT
            
            else:  # cat's turn
                for dr, dc in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                    for k in range(cat_jump + 1):
                        nr = cr + dr * k
                        nc = cc + dc * k
                        if not (0 <= nr < R and 0 <= nc < C):
                            break
                        if A[nr][nc] == '#':
                            break
                        if decide(mr, mc, nr, nc, turn + 1) == CAT:
                            return CAT
                return MOUSE

        return decide(MR, MC, CR, CC, 0) == MOUSE
Alex Wice — 02/11/2021
Find the Highest Altitude

Find all heights, which you can do with a running total.  The answer is the maximum height ever seen.

class Solution:
    def largestAltitude(self, gain):
        ans = h = 0
        for delta in gain:
            h += delta
            ans = max(ans, h)
        return ans
Minimum Number of People to Teach

For each edge ("friendship"), we only care about that edge if it is "bad", ie. if the two people can't communicate.

Now consider if we teach some language lang to a set of people.  For each bad edge (u, v), we need to teach u and v depending on if they know lang or not.  Afterwards, a candidate answer is the number of people we needed to teach.

class Solution:
    def minimumTeachings(self, n, langs, edges):
        m = len(langs)
        langsets = [set(row) for row in langs]
        ans = math.inf
        
        badedges = []  # 0 indexed - edges that dont have common language
        for u, v in edges:
            u -= 1
            v -= 1
            if not langsets[u] & langsets[v]:
                badedges.append([u, v])

        for lang in set.union(*langsets):
            needs = [False] * m
            
            for u, v in badedges:
                if lang not in langsets[u]:
                    needs[u] = True
                if lang not in langsets[v]:
                    needs[v] = True
            
            ans = min(ans, sum(needs))
        
        return ans
Decode XORed Permutation

A key fact is that S = 1 XOR 2 XOR ... XOR N is 1 if N % 4 == 1, and 0 if N % 4 == 3.

Now encoded[1] XOR encoded[3] XOR encoded[5] ... must be S XOR ans[0].

From this, you can get ans[0], and the result follows.

class Solution:
    def decode(self, encoded):
        n = len(encoded) + 1
        a = 1 if n % 4 == 1 else 0
        for x in encoded[1::2]:
            a ^= x
        
        ans = [a]
        for x in encoded:
            ans.append(ans[-1] ^ x)
        
        return ans
Count Ways to Make Array With Product

Let's solve each query (n, p) separately.

Say n = (p1 ** e1) * (p2 ** e2) * ... where p_i are primes.  Evidently, we only care about the sequence (e1, e2, ...) in forming the answer.

For each exponent e_i, let's figure out how many ways there are to distribute the factors of p_i ** e_i.  Let's say it occupies d positions in the array.  We choose these positions in binom(n, d) ways, then distribute them in partition(e - d, d) ways (as each position must get atleast one factor).  partitions(n, k) = binom(n + k - 1, k - 1) based on a so-called "stars and bars" argument (you can search this on Google.)


MOD = 10 ** 9 + 7
fac = [1] * (10 ** 4 + 1)
for i in range(2, 10 ** 4 + 1):
    fac[i] = fac[i - 1] * i % MOD

ifac = fac[:]
ifac[-1] = pow(fac[-1], MOD - 2, MOD)
for i in range(10 ** 4 - 1, -1, -1):
    ifac[i] = ifac[i + 1] * (i + 1) % MOD

class Solution:
    def waysToFillArray(self, queries: List[List[int]]) -> List[int]:
        def binom(n, k):
            return fac[n] * ifac[n - k] % MOD * ifac[k] % MOD

        def partition(n, k):
            return binom(n + k - 1, k - 1)

        def solve(n, p):
            expos = []
            d = 2
            while d * d <= p:
                e = 0
                while p % d == 0:
                    p //= d
                    e += 1
                if e:
                    expos.append(e)
                
                d += 1 + (d & 1)
            
            if p > 1:
                expos.append(1)
            
            ans = 1
            for e in expos:
                ways = 0
                for d in range(1, min(n, e) + 1):
                    ways += binom(n, d) * partition(e - d, d)
                ans *= ways
                ans %= MOD
            
            return ans

        return [solve(*query) for query in queries]
Latest Time by Replacing Hidden Digits

We can do the hours and the minutes separately.

For the hours, we try numbers in descending order.  If it matches the template, we break.  The minutes case is similar.

class Solution(object):
    def maximumTime(self, time):
        def match(x, y):
            return x == y or x == '?'

        for h in range(23, -1, -1):
            s = str(h).zfill(2)
            if match(time[0], s[0]) and match(time[1], s[1]):
                break
        
        for m in range(59, -1, -1):
            s = str(m).zfill(2)
            if match(time[3], s[0]) and match(time[4], s[1]):
                break
        
        return "{:02}:{:02}".format(h, m)
Change Minimum Characters to Satisfy One of Three Conditions

All that matters in the strings is the frequency of each letter.

Now we proceed in cases: either everything is equal to i, or a <= i and b > i, or a > i and b <= i.

class Solution(object):
    def minCharacters(self, a, b):
        counta = [0] * 26
        countb = [0] * 26
        for c in a:
            counta[ord(c) - ord('a')] += 1
        for c in b:
            countb[ord(c) - ord('a')] += 1
        
        ans = n = len(a) + len(b)
        
        # Case 1: all equal to i
        for i in range(26):
            ans = min(ans, n - counta[i] - countb[i])
        
        # Case 2: a <= i, b > i
        for i in range(25):
            good = sum(counta[c] for c in range(i + 1))
            good += sum(countb[c] for c in range(i + 1, 26))
            ans = min(ans, n - good)
        
        # Case 3: a > i, b <= i
        for i in range(25):
            good = sum(countb[c] for c in range(i + 1))
            good += sum(counta[c] for c in range(i + 1, 26))
            ans = min(ans, n - good)
        
        return ans
Find Kth Largest XOR Coordinate Value

Using the same ideas as 2D prefix sums, we can create a new array dp[][] which has dp[r][c] as the "value" of (r, c) as described in the problem.

Then, we make a list of all the values and choose the K-th highest.

class Solution:
    def kthLargestValue(self, A, K):
        R, C = len(A), len(A[0])
        dp = [[0] * (C + 1) for _ in range(R + 1)]
        for r, row in enumerate(A):
            for c, v in enumerate(row):
                dp[r+1][c+1] = dp[r][c+1] ^ dp[r+1][c] ^ v ^ dp[r][c]
        
        vals = []
        for row in dp:
            vals.extend(row)
        vals.sort()
        return vals[-K]
Alex Wice — 02/11/2021
Building Boxes

The shape you can make resembles the plane x + y + z = c intersecting the first octet of R^3.

Imagine building this one xz layer at a time, instead of xy.  On the i-th layer, you can put down 1 block and ans++, then 2 blocks and ans++, and so on until you put down i blocks.

class Solution(object):
    def minimumBoxes(self, n):
        ans = 0
        for i in range(1, 2000):
            for j in range(1, i + 1):
                if n <= 0:
                    break
                ans += 1
                n -= j
        return ans
Maximum Number of Balls in a Box

For each ball numbered x, find the box s that it is in, and add that to the count.

class Solution:
    def countBalls(self, lo, hi):
        count = Counter()
        for x in range(lo, hi + 1):
            s = sum(map(int, str(x)))
            count[s] += 1
        
        return max(count.values())
Restore the Array From Adjacent Pairs

Build the graph from the adjacent edges.  Now two nodes have degree one (the two endpoints of the array).  You can build the array left to right once you know some endpoint u.

class Solution(object):
    def restoreArray(self, edges):
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        for u in graph:
            if len(graph[u]) == 1:
                break

        ans = [u]
        while len(ans) < n:
            for v in graph[ans[-1]]:
                if len(ans) == 1 or v != ans[-2]:
                    ans.append(v)
                    break
        
        return ans
Can You Eat Your Favorite Candy on Your Favorite Day?

Let's answer each query individually.  Looking at the slowest and fastest possible time we could eat candy, we can figure out whether it is possible (as any rate between slowest and fastest is possible.)

class Solution:
    def canEat(self, A, queries):
        P = list(accumulate(A, initial=0))
        
        def query(typ, day, cap):
            # P[typ + 1] : all candies including favorite
            # P[typ] + 1 : all candies before the favorite, plus 1 of the favorite
            
            # note1: eating slow, must have at least 1 fave candy left
            # note2: eating fast, must be able to eat 1 of the fave candy in time
            return (
                (day + 1) <= P[typ + 1] and     # note1
                (day + 1) * cap >= P[typ] + 1   # note2
            )
        
        return [query(*q) for q in queries]
Palindrome Partitioning IV

Let isp(i, j) be true iff s[i..j] is a palindrome.  We can find it with a simple dp.

Now for each i <= j, we can just check if isp(0, i-1) and isp(i, j) and isp(j+1, n-1).

class Solution:
    def checkPartitioning(self, s: str) -> bool:
        @cache
        def isp(i, j):
            if i >= j:
                return True
            return s[i] == s[j] and isp(i + 1, j - 1)
        
        n = len(s)
        for i in range(1, n):
            for j in range(i, n - 1):
                if isp(0, i-1) and isp(i, j) and isp(j+1, n-1):
                    return True
        return False
Sum of Unique Elements

To determine whether a value is unique, use a collections.Counter.

class Solution:
    def sumOfUnique(self, nums):
        count = Counter(nums)
        return sum(x for x in count if count[x] == 1)
Maximum Absolute Sum of Any Subarray

Similar to Kadane's Algorithm, we keep track of the largest subarray sum ending in i (call this hi), as well as the smallest (call this lo).  Then the answer is the max of some hi or -lo.

class Solution:
    def maxAbsoluteSum(self, A):
        ans = hi = lo = 0
        for x in A:
            hi = max(x, hi + x)
            lo = min(x, lo + x)
            ans = max(ans, hi, -lo)
        return ans
Minimum Length of String After Deleting Similar Ends

Two pointer.  i, j represent the string s[i..j].  While s[i] == s[j], we move i and j to positions that are indices of new characters.

class Solution:
    def minimumLength(self, s: str) -> int:
        i = 0
        j = len(s) - 1
        while i < j and s[i] == s[j]:
            c = s[i]
            while i < j and s[i] == c:
                i += 1
            while i <= j and s[j] == c:
                j -= 1
        
        return j - i + 1
Maximum Number of Events That Can Be Attended II

DP.  Sort the events, and let dp(i, rem) be the answer for A[i:] where you attend rem of them.

If you don't go to the event, you get dp(i+1, rem) points.  If you do go to the event A[i] = [s, e, v], then you get v + dp(j, rem - 1) points, where j is the first index such that A[j][0] > e.  We can binary search for such an index.

class Solution:
    def maxValue(self, A, K):
        A.sort()
        
        @functools.cache
        def dp(i, rem):
            if i >= len(A) or rem == 0:
                return 0
            
            j = bisect_right(A, [A[i][1], math.inf])
            take = A[i][2] + dp(j, rem - 1)
            return max(dp(i + 1, rem), take)
        
        return dp(0, K)
Check if Array Is Sorted and Rotated

In a brute force solution, we have some ideal (sorted) array, and we rotate the array n times and check whether it equals the ideal array.

In the efficient solution, we know where to rotate based on the differences between adjacent elements - the biggest difference must be between the end and the beginning of the array.  After, we check whether this rotation is monotone increasing.

class Solution:
    def check(self, nums: List[int]) -> bool:
        # O(n) solution
        deltas = [nums[i] - nums[i-1] for i in range(len(nums))]
        i = deltas.index(min(deltas))
        # Is nums[i:] + nums[:i] monotone increasing?
        n = len(nums)
        return all(nums[(i + j) % n] <= nums[(i + j + 1) % n] for j in range(n - 1))
        
        # Brute force (O(n^2))
        ideal = sorted(nums)
        for i in range(len(nums)):
            rotated = nums[i:] + nums[:i]
            if rotated == ideal:
                return True
        return False
Maximum Score From Removing Stones

Without loss of generality, suppose a <= b <= c.  You can always make a move unless c > a+b, and this is the only condition that undermines your progress.  So basically, set c = min(c, a+b) and then it is greedy.

class Solution:
    def maximumScore(self, a: int, b: int, c: int) -> int:
        a, b, c = sorted([a, b, c])
        c = min(c, a + b)
        return a + b + c >> 1
Largest Merge Of Two Strings

To know whether to take s[0] or t[0] next, it turns out to be dependent on whether s >= t or not.

class Solution:
    def largestMerge(self, s, t):
        ans = []
        while s or t:
            if s >= t:
                ans.append(s[0])
                s = s[1:]
            else:
                ans.append(t[0])
                t = t[1:]
        
        ans.extend(s)
        ans.extend(t)
        return "".join(ans)
Closest Subsequence Sum

Meet in the middle.  Split array A into halves A = L + R, and let left, right be sorted arrays of the sum of every subset of L and R respectively.

Now we can use two pointers to find the answer.  For each x in left, there are two candidates right[j] and right[j+1] that would have x + right[j] be close to target.

class Solution(object):
    def minAbsDifference(self, A, target):
        N = len(A)
        def make(B):
            seen = {0}
            for x in B:
                seen |= {y + x for y in seen}
            return sorted(seen)
        
        left = make(A[: N >> 1])
        right = make(A[N >> 1 :])
        
        ans = float('inf')
        j = len(right) - 1
        for i, x in enumerate(left):
            while j >= 0 and x + right[j] > target:
                j -= 1
            
            if j >= 0:
                ans = min(ans, abs(x + right[j] - target))
            if j + 1 < len(right):
                ans = min(ans, abs(x + right[j + 1] - target))
        
        return ans