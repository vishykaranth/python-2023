class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        left = 0
        dummy = cur = ListNode(-1)
        while l1 or l2 or left:
            left, sm = divmod(sum(l and l.val or 0 for l in (l1, l2)) + left, 10)
            cur.next = cur = ListNode(sm)
            l1 = l1 and l1.next
            l2 = l2 and l2.next
        return dummy.next


# Test case
def create_linked_list(values):
    dummy = current = ListNode(-1)
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next


def print_linked_list(node):
    while node:
        print(node.val, end=" -> ")
        node = node.next
    print("None")


def print_linked_list_reverse(node):
    values = []
    while node:
        values.append(node.val)
        node = node.next

    values.reverse()

    for value in values:
        print(value, end=" -> ")
    print("None")


# Example usage:
l1 = create_linked_list([2, 4, 3])  # Represents the number 342
l2 = create_linked_list([5, 6, 4])  # Represents the number 465

solution = Solution()
result = solution.addTwoNumbers(l1, l2)

print("Input Linked List 1:")
print_linked_list_reverse(l1)

print("Input Linked List 2:")
print_linked_list_reverse(l2)

print("Resulting Linked List:")
print_linked_list_reverse(result)
