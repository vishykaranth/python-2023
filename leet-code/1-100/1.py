class Solution:
    def twoSum(self, nums, target):
        nums_hash = {}
        for i in range(len(nums)):
            if target - nums[i] in nums_hash:
                return [nums_hash[target - nums[i]], i]
            nums_hash[nums[i]] = i


s = Solution()

print(s.twoSum([1, 5, 7, 8], 15))

# self, range