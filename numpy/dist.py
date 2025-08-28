import numpy as np

def dist_between_pairs(*li):
    print("     ", end="")
    for i in range(len(li)):
        print(f"  {i}  ", end=" | ")
    print("")
    for i in range(len(li)):
        print(f"[{i}]: ", end="")
        for j in range(len(li)):
            print(f"{np.linalg.norm(li[i] - li[j]):.3f}", end=" | ")
        print("")
