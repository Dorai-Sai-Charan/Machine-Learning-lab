def Euclidean_dist(vect1,vect2):
    
    if(len(vect1)!=len(vect2)):
        print("Please provide the lengths of the vectors as same")
        
    eucdist=0
    for i in range(len(vect1)):
        eucdist+=(vect1[i]-vect2[i])**2
        
    return eucdist**0.5

def Manhattan_dist(vect1,vect2):
    
    if(len(vect1)!=len(vect2)):
        print("Please provide the lengths of the vectors as same")
        
    mandist=0
    for i in range(len(vect1)):
        mandist+=abs(vect1[i]-vect2[i])
        
    return mandist

vect1=[1,2,3]
vect2=[4,5,6]

print("The euclidean distance of the given vectors is",Euclidean_dist(vect1,vect2))
print("The Manhattan distance of the given vectors is",Manhattan_dist(vect1,vect2))