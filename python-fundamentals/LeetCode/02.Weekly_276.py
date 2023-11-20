import functools
from typing import List


def divideString(s: str, k: int, fill: str) -> List[str]:
    to_ret = []
    while len(s) > 0:
        to_ret.append(s[:k])
        s = s[k:]
    to_ret[-1] += fill * (k - len(to_ret[-1]))
    return to_ret


def minMoves(target: int, maxDoubles: int) -> int:
    to_ret = 0
    while target > 1:
        if maxDoubles == 0:
            to_ret += target - 1
            break
        if target % 2 == 1:
            to_ret += 1
            target -= 1
        else:
            to_ret += 1
            target = target // 2
            maxDoubles -= 1
    return to_ret


def mostPoints(questions: List[List[int]]) -> int:
    @functools.lru_cache(None)
    def solve(t=0):
        if t >= len(questions):
            return 0
        points, brainpower = questions[t]
        return max(points + solve(t + brainpower + 1), solve(t + 1))

    return solve()


def maxRunTime(n: int, batteries: List[int]) -> int:
    batteries = sorted(batteries, reverse=True)
    sumt = sum(batteries)
    for t in batteries:
        if t > sumt // n:
            n -= 1
            sumt -= t
        else:
            return sumt // n
