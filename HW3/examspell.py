from dis import dis

# from spellchecker import edit_distance


def dist(m,c):
    len_m = len(m)
    len_c = len(c)
    dist_matr = [([0]*(len_c + 1)) for i in range(len_m+1)]
    
    dist_matr[0][0]= 0
    for mi in range (len(dist_matr)):
        dist_matr[mi][0]= mi
    for ci in range (len(dist_matr[0])):
        dist_matr[0][ci] = ci
    
    for mi in range(1, len(dist_matr)):
        for ci in range(1, len(dist_matr[0])):
            sub_c = dist_matr[mi-1][ci-1] + cost_sub(m[mi-1], c[ci-1])
            ins_c = dist_matr[mi][ci-1] + cost_ins(c[ci-1])
            del_c = dist_matr[mi-1][ci] + cost_del(m[mi-1])
            print("current char:" + m[mi-1])
            # print("current char:" + m[mi-2])
            dist_matr[mi][ci] = min(sub_c, ins_c, del_c)
    return dist_matr[len_m][len_c]

def cost_ins(ch):
    return 1

def cost_del(ch):
    cost = 1
    list = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
    if (ch in list):
        cost = 0.5
    return cost

def cost_sub(ch_m, ch_c):
    if ch_m == ch_c:
        return 0
    else:
        return 2
    
def main():
    wrong1 = "herllo"
    right1 = "hello"
    wrong2 = "swewxcy"
    right2 = "sexy"
    right3 = "swepp"
    print(dist(wrong2,right2))
    
    

if __name__ == "__main__":
    main()