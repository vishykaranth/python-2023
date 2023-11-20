from collections import Counter

from urllib3.connectionpool import xrange


def countElements(nums):
    a = nums
    lo = min(a)
    hi = max(a)
    return sum(1 for v in nums if lo < v < hi)


print(countElements([11, 7, 2, 15]))
print(countElements([-3, 3, 3, 90]))


def rearrangeArray(nums):
    r = [0] * len(nums)
    r[::2] = [v for v in nums if v > 0]
    r[1::2] = [v for v in nums if v < 0]
    return r


print(rearrangeArray([3, 1, -2, -5, 2, -4]))
print(rearrangeArray([-1, 1]))


def findLonely(nums):
    c = Counter(nums)
    return [k for k, v in c.items()
            if v == 1 and (k - 1) not in c and (k + 1) not in c]


print(findLonely([10, 6, 5, 8]))
print(findLonely([10, 6, 5, 9]))
print(findLonely([1, 3, 5, 3]))


# print(Counter('abcdeabcdabcaba').most_common(3))

def maximumGood(statements):
    a = statements
    n = len(a)
    m = 1 << n
    pc = [0] * m
    for i in xrange(1, m):
        pc[i] = pc[i & (i - 1)] + 1

    def _check(msk):
        t = [(msk >> i) & 1 for i in xrange(n)]
        for i in xrange(n):
            if not t[i]:  # not trustworthy
                continue
            for j in xrange(n):
                if a[i][j] != 2 and t[j] != a[i][j]:
                    return False
        return True

    return max(pc[msk] for msk in xrange(m) if _check(msk))


print(maximumGood([[2, 1, 2], [1, 2, 2], [2, 0, 2]]))
print(maximumGood([[2, 0], [0, 2]]))
