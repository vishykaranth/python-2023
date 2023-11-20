class Solution:
    def isAnagram(self, s, t):
        return sorted(s) == sorted(t)



# The time complexity of the provided `isAnagram` function can be analyzed by looking at the dominant operations. In this case, the dominant operations are the sorting of the input strings `s` and `t`.
#
# Let \(n\) be the length of the longer of the two input strings (either `s` or `t`). The time complexity of sorting a list of length \(n\) using a comparison-based sorting algorithm (such as the one used in Python's `sorted` function) is typically \(O(n \log n)\).
#
# Since the `sorted` function is applied to both strings `s` and `t`, the overall time complexity of the `isAnagram` function is \(O(n \log n)\), where \(n\) is the length of the longer of the two input strings.