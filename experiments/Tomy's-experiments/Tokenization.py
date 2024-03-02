
# initializing list
test_list = ["Testing for tokenization", "Fat cat syndrome", "Test 01"]

# printing original list
print("The original list : " + str(test_list))
# List splitting
res = [sub.split() for sub in test_list]

# print result
print("The list after split of strings is : " + str(res))
