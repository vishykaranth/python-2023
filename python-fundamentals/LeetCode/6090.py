from typing import List


def process(nums: List[int]) -> List[int]:
    if len(nums) == 1:
        return nums[0]
    result = []
    newIndex = 0
    idx = 0
    while len(nums) > idx:
        if newIndex % 2 == 0:
            val = min(nums[idx], nums[idx + 1])
        else:
            val = max(nums[idx], nums[idx + 1])
        result.append(val)
        newIndex = newIndex + 1
        idx = idx + 2

    return result

def minMaxGame(nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    while len(nums) > 1:
        nums = process(nums)

    return nums


print(minMaxGame([1, 3, 5, 2, 4, 8, 2, 2]))
# print(process([1, 3, 5, 2]))
# print(process([1, 3, 5, 2, 4, 8, 2, 2]))
# print(process([1, 5, 4, 2]))
# print(process([1, 4]))
# print(process([3]))
