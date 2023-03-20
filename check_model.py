
with open('model_new.txt','r') as new:
    n_ = new.readlines()

with open('model_old.txt','r') as old:
    o_ = old.readlines() 

for i in range(len(o_)):
    ln = n_[i]
    lo = o_[i]
    if ln[0] == '-':
        print(ln)
    if ln != lo:
        print(f"Difference in line {i}: New: {ln} Old: {lo}")
        break
