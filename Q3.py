def commondigits(in_list1,in_list2):
    list1=set(in_list1)#every list is converted to set so that we can use intersection function of sets
    list2=set(in_list2)
    
    common_digits=list1.intersection(list2)
    
    return len(common_digits)

list1=[5,4,3,6,7]
list2=[1,2,3,4,5]

print("the first list is",list1)
print("The secnd list is",list2)

common_digit_count=commondigits(list1,list2)

print("The common elements",common_digit_count)