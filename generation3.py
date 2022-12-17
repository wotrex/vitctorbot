import numpy as np
from scipy.spatial import Voronoi#, voronoi_plot_2d
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random as rnd
from perlin import PerlinNoiseFactory
import math
import time
from RegionClass import Region
import json
import pymongo
import io

def per_gen(size,frames,res, frameres = 1): ##Генерація шумів
    space_range = (size+res)//2
    frame_range = frames//frameres
    pnf = PerlinNoiseFactory(3, octaves=4, tile=(space_range, space_range, frame_range))
    arr_noisy = np.zeros((frames, size), dtype=int)
    for t in range(frames):
        for x in range(size):
            n = pnf(x/res, x/res, t/frameres)
            s = int((n + 1) / 2 * 255 + 0.5)
            arr_noisy[t][x] = s
    return arr_noisy



def draw_line(x1,y1,x2,y2,draw, color, coef, m): ## Отрисовка кривих ліній

    full_len = math.sqrt(np.power((x2-x1),2) + np.power((y2-y1),2)) # Находим длину линии
    noise = per_gen(int(full_len)+2,1,full_len*m)[0] # Генерируем шум Перлинга

    for i in range(int(len(noise)*0.1)): # Переделываем последнюю часть масива для логического завершения линии
        noise[-i-1] = noise[i]

    lx = x1 # Треба для дорисовки пробелов
    ly = y1 #
    for lenn in range(len(noise)): # Находим каждую точку линии и сдвигаем по шуму
        offst = (noise[lenn]-128)//coef
        k = lenn/full_len
        x = (x1 +(x2-x1) * k)
        y = (y1 +(y2-y1) * k)

        vect = (x-x2, y-y2) # Получаем вектор текущей и конечной точек
        len_pnt = math.sqrt(np.power(vect[0],2) + np.power(vect[1],2)) #Находим длину между точками
        vect = (vect[1] / len_pnt, (vect[0] * -1 )/len_pnt) #Делаєм вектор перпендикуляр и нормализуем(Деление на длину)
        vect = (vect[0] * offst,vect[1] * offst)#Умножаем на нужную длину
        x += vect[0] # Получаем нужную точку
        y += vect[1]
        draw.line(xy=((x, y),(lx, ly),), fill=color, width=1)
        lx=x
        ly=y



##########   Start ############

def generate(chatid):
    db_client = pymongo.MongoClient("mongodb+srv://wotrex:i98wbdz9@victorgame.2kqtytt.mongodb.net/?retryWrites=true&w=majority")
    curtime = time.time()

    size_x= 3840 #3840 
    size_y= 2880 #2160  #Размер мапы(изображения)
    to_json = {'size': [size_x,size_y]}
    to_json['id'] = chatid
    points = np.array([[rnd.randint(0,size_x), rnd.randint(0,size_y)]])
    count= rnd.randint(5000,6500) #Количество регионов
    for i in range(count-1):
        x = rnd.randint(0,size_x)
        y = rnd.randint(0,size_y)
        points = np.append(points, [[x,y]], axis=0)
    vor = Voronoi(points) #Регионы до релаксации
    print('Створюємо точки.........')
    #Релаксация Ллойда
    k = 2 # Количество кругов релаксация(опционально)
    new_points = np.copy(points)
    for l in range(k):
        new_vor = Voronoi(new_points)
        new_points = []
        nreg = new_vor.regions
        nregid = new_vor.point_region
        nver = new_vor.vertices
        for r in nregid:
            if len(nreg[r])==0 or -1 in nreg[r]:
                continue
            sum_x = 0
            sum_y = 0
            for i in nreg[r]:
                sum_x += nver[i][0]
                sum_y += nver[i][1]
            new_points.append([sum_x/len(nreg[r]),sum_y/len(nreg[r])])
    new_points = np.array(new_points)
    to_json['points'] = new_points.tolist()

    #Точки для рисовки
    n_vor = Voronoi(new_points) #Регионы после релаксации
    reg = vor.regions # Регионы с индексами вершин
    ver = vor.vertices # Координаты вершин
    n_reg = n_vor.regions
    n_ver = n_vor.vertices
    n_regid = n_vor.point_region
    to_json['regid'] = n_regid.tolist()
    to_json['reg'] = n_reg
    to_json['ver'] = n_ver.tolist()



    ##fig = voronoi_plot_2d(vor) # отобразить диаграму 
    print('Створюємо регіони.........')
    #Создаем и сортируем регионы

    regions = np.empty(len(new_points) , dtype=Region)
    for r in range(len(new_points)):
        region = Region(r, new_points[r].tolist())
        try:
            n_reg[n_regid[r]].remove(-1)
        except:
            pass
        region.points = n_reg[n_regid[r]]
        for p in range(len(n_reg[n_regid[r]])):
            v = [n_ver[n_reg[n_regid[r]][p]][0],n_ver[n_reg[n_regid[r]][p]][1]]
            i = [n_ver[n_reg[n_regid[r]][p-1]][0],n_ver[n_reg[n_regid[r]][p-1]][1]]
            for rg in regions:
                if rg == None:
                    break
                if v in rg.points_cord and i in rg.points_cord:
                    region.neighbors.append(rg.id)
                    rg.neighbors.append(region.id)
            region.points_cord.append(v)

        regions[r] = region
    print('Розприділяємо регіони на воду і сушу.........')
    # Распределяем на воду и сушу
    count_lands = rnd.randint(int(len(new_points)*0.4),int(len(new_points)*0.55)) # Указываем количество суши
    lands = []
    new_lnd = []
    start_points = rnd.randint(9,18)   # Берётся несколько рандомных точек. Точки и их соседи становятся сушей
    for l in range(start_points):
        rg = rnd.randint(1,len(regions))
        regions[rg].type = -1
        lands.append(rg)
        for n in regions[rg].neighbors:
            regions[n].type = -1
            new_lnd.append(n)

    while len(lands) + len(new_lnd) < count_lands:         # Берутся точка, которая еще не бралась и которае уже являеться сушей. Её соседи становятся сушей
        r = rnd.choice(new_lnd)                            # И так дальше, пока не наберётся нужное количество суши
        for n in regions[r].neighbors:
            if regions[n].type == 0:
                regions[n].type = -1
                new_lnd.append(n)

        new_lnd.remove(r)
        lands.append(r)

    lands += new_lnd
    waters = []
    print('Кількість регіонів:',len(new_points))
    to_json['count_reg'] = len(new_points)
    for i in range(len(new_points)):
        if i in lands:
            continue
        waters.append(i)


    print('Створюємо острова.........')
    #Добавляем острова
    count_islands = rnd.randint(15,25)   # Количество островов
    for i in range(count_islands):            # Берём рандомную точку на воде и делаем из неё сушу. 50 на 50 что соседний регион тоже будет сушей
        r = rnd.choice(waters)
        regions[r].type = -1
        for n in regions[r].neighbors:
            if regions[n].type == 0:
                rand = rnd.randint(0,2)
                if rand == 1:
                    regions[n].type = -1
                    lands.append(n)
                    waters.remove(n)
        lands.append(r)
        waters.remove(r)
    new_lnd = lands.copy()

    odd_regs = set(lands).intersection(waters) #Убираем избыточные регионы
    odd_regs = list(odd_regs)
    print(odd_regs)
    for i in odd_regs:
        waters.remove(i)

    lands = np.array(lands)
    waters = np.array(waters)

    print('Розставляємо тип місцевості.........')
        ## Тип местности
    height = per_gen(len(lands),1,len(lands) * 0.01)[0]
    wetness = per_gen(len(lands),1,len(lands) * 0.01)[0]
    # for r in range(len(lands)):

    # Генерируем шум для влажности и высоты
    # Тип местности будет зависить от высоты, влажности и расположения по оси Y 
    # Соседние регионы также получают такой же тип местности
    cr = 0 
    mountains = []
    while len(new_lnd) != 0:
        ###
        regions[new_lnd[0]].height = int(height[cr])
        regions[new_lnd[0]].wetness = int(wetness[cr])
        if regions[new_lnd[0]].centr[1] < size_y * 0.3:                    ### 
            if regions[new_lnd[0]].centr[1] < size_y * 0.2:
                regions[new_lnd[0]].temperature = -1
            else:
                rand1 = rnd.randint(0,1)
                if rand1 == 1:
                    regions[new_lnd[0]].temperature = -1
        if regions[new_lnd[0]].centr[1] > size_y * 0.7:                    ##Определяем температуру
            if regions[new_lnd[0]].centr[1] > size_y * 0.8:
                regions[new_lnd[0]].temperature = 1
            else:
                rand1 = rnd.randint(0,1)
                if rand1 == 1:
                    regions[new_lnd[0]].temperature = 1                    ## 
        # Определяем тип местности
        if height[cr] < 140:
            if regions[new_lnd[0]].temperature == 1:
                if wetness[cr] <= 125:
                    regions[new_lnd[0]].type = 12
                else:
                    regions[new_lnd[0]].type = 11
            if regions[new_lnd[0]].temperature == 0:
                if wetness[cr] <= 155:
                    if wetness[cr] <= 140:
                        if wetness[cr] <= 130:
                            if wetness[cr] <= 115:
                                regions[new_lnd[0]].type = 9
                            else:
                                regions[new_lnd[0]].type = 3
                        else:
                            regions[new_lnd[0]].type = 8
                    else:
                        regions[new_lnd[0]].type = 7
                else:
                    regions[new_lnd[0]].type = 6
            if regions[new_lnd[0]].temperature == -1:
                if wetness[cr] <= 130:
                    if wetness[cr] <= 100:
                        regions[new_lnd[0]].type = 5
                    else:
                        regions[new_lnd[0]].type = 3
                else:
                    regions[new_lnd[0]].type = 4
        else:
            if height[cr] > 147:
                regions[new_lnd[0]].type = 1
            else:
                regions[new_lnd[0]].type = 2
        ###
        for n in regions[new_lnd[0]].neighbors:
            if regions[n].type == 0:
                continue
            if n in new_lnd:
                if height[cr] > 139:
                    rand1 = rnd.randint(0,1)
                    mountains.append(new_lnd[0])
                    if rand1 == 0:
                        continue
                    else:
                        mountains.append(n)
                if regions[new_lnd[0]].type == 3:
                    rand1 = rnd.randint(0,1)
                    if rand1 == 0:
                        continue
                regions[n].height = int(height[cr])
                regions[n].wetness = int(wetness[cr])
                regions[n].temperature = regions[new_lnd[0]].temperature
                regions[n].type = regions[new_lnd[0]].type
                new_lnd.pop(new_lnd.index(n))
        new_lnd.pop(0)
        cr += 1


    ###Горные хребты
    count_hrebets = rnd.randint(8,14)
    hrebets = 0

    while hrebets < count_hrebets:
        r = rnd.choice(mountains)
        vec_x = rnd.randint(-1,1) 
        vec_y = rnd.randint(-1,1)
        if vec_x == 0 and vec_y == 0:
            vec_x = 1
        mount_len = rnd.randint(6,13)
        while mount_len > 1:
            pnts = []
            for n in regions[r].neighbors:
                if regions[n].type == 0:
                    continue
                pnts.append(n)
            if len(pnts) == 0:
                break
            pnt = []
            for p in pnts:
                if vec_x == 1 and vec_y ==0:
                    if regions[p].centr[0] >= regions[r].centr[0]:
                        pnt.append(p)
                if vec_x == 0 and vec_y ==1:
                    if regions[p].centr[1] >= regions[r].centr[1]:
                        pnt.append(p)
                if vec_x == 0 and vec_y == -1:
                    if regions[p].centr[1] <= regions[r].centr[1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==0:
                    if regions[p].centr[0] <= regions[r].centr[0]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==1:
                    if regions[p].centr[0] >= regions[r].centr[0] and regions[p].centr[1] >= regions[r].centr[1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==-1:
                    if regions[p].centr[0] <= regions[r].centr[0] and regions[p].centr[1] <= regions[r].centr[1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==1:
                    if regions[p].centr[0] <= regions[r].centr[0] and regions[p].centr[1] >= regions[r].centr[1]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==-1:
                    if regions[p].centr[0] >= regions[r].centr[0] and regions[p].centr[1] <= regions[r].centr[1]:
                        pnt.append(p)
            if len(pnt) == 0:
                break
            p = rnd.choice(pnt)
            regions[p].type = 1
            regions[p].height = 150
            r = p
            mount_len -= 1
        hrebets += 1



    print('Створюємо річки.........')
    ## Делаем реки
    count_rivers = rnd.randint(50,80)
    rivers = []
    lands_for_river = []
    current_points = []
    for l in lands:
        if regions[l].type == 1 or regions[l].type == 11 or regions[l].type == 12:
            continue
        skip = False
        for r in regions[l].neighbors:
            if regions[r].type == 0:
                skip = True
                break
        if skip:
            continue
        lands_for_river.append(l)

    while len(rivers) < count_rivers:     # Берётся участок суши без соседней воды и от него по соседним точкам идёт река, пока не достигнет моря
        # len_river = rnd.randint(20,30)
        current_pnt = []
        river = []
        c = 0
        if len(lands_for_river) == 0:
            break
        r = rnd.choice(lands_for_river)
        cur_r = r
        lands_for_river.remove(cur_r)
        vec_x = rnd.randint(-1,1) 
        vec_y = rnd.randint(-1,1)
        if vec_x == 0 and vec_y == 0:
            vec_x = 1 
        start_pnt = rnd.choice(regions[r].points)
        river.append(start_pnt)
        current_points.append(start_pnt)
        regions[r].rivers = True
        wh = True
        while wh:
            neighbors = []
            neighbor  = 0
            pnts = []
            for n in regions[r].neighbors:
                for p in regions[n].points:
                    ind = -1
                    if p == river[-1]:
                        ind = regions[n].points.index(p)
                    if ind == -1:
                        continue
                    if regions[n].points[ind-1] not in current_points and regions[n].points[ind-1] not in current_pnt:
                        pnts.append(regions[n].points[ind-1])
                        neighbors.append(n)
                    try:
                        if regions[n].points[ind+1] not in current_points and regions[n].points[ind+1] not in current_pnt:
                            pnts.append(regions[n].points[ind+1])
                            neighbors.append(n)
                    except:
                        pass
            if len(pnts) == 0:
                r = cur_r
                river = river[:1]
                vec_x = rnd.randint(-1,1) 
                vec_y = rnd.randint(-1,1)
                if vec_x == 0 and vec_y == 0:
                    vec_x = 1 
                if c == 8:
                    break
                c += 1
                continue
            pnt = []
            for p in pnts:
                if vec_x == 1 and vec_y ==0:
                    if n_ver[p][0] >= n_ver[river[-1]][0]:
                        pnt.append(p)
                if vec_x == 0 and vec_y ==1:
                    if n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == 0 and vec_y == -1:
                    if n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==0:
                    if n_ver[p][0] <= n_ver[river[-1]][0]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==1:
                    if n_ver[p][0] >= n_ver[river[-1]][0] and n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==-1:
                    if n_ver[p][0] <= n_ver[river[-1]][0] and n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==1:
                    if n_ver[p][0] <= n_ver[river[-1]][0] and n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==-1:
                    if n_ver[p][0] >= n_ver[river[-1]][0] and n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)

            if len(pnt) == 0:
                r = cur_r
                river = river[:1]
                vec_x = rnd.randint(-1,1) 
                vec_y = rnd.randint(-1,1)
                if vec_x == 0 and vec_y == 0:
                    vec_x = 1 
                if c == 8:
                    break
                c += 1
                continue
            p = rnd.choice(pnt)
            for n in neighbors:
                if p in regions[n].points:
                    neighbor = n
            if regions[neighbor].type == 0:
                break
            river.append(p)
            current_pnt.append(p)
            r = neighbor

        if len(river) > 2:
            current_points += current_pnt
            rivers.append(river)

    count_add_rivers = len(rivers)//3
    add_rivers = []
    while len(add_rivers) < count_add_rivers:
        current_pnt = []
        len_river = rnd.randint(5,15)
        river = []
        c = rnd.choice(rivers)
        c = rnd.choice(c)
        r = 0
        for i in lands:
            if c in regions[i].points:
                r = i
                break
        cur_r = r
        vec_x = rnd.randint(-1,1) 
        vec_y = rnd.randint(-1,1)
        if vec_x == 0 and vec_y == 0:
            vec_x = 1 
        start_pnt = c
        river.append(start_pnt)
        current_points.append(start_pnt)
        regions[r].rivers = True
        while len(river) < len_river:
            neighbors = []
            neighbor  = 0
            pnts = []
            for n in regions[r].neighbors:
                for p in regions[n].points:
                    ind = -1
                    if p == river[-1]:
                        ind = regions[n].points.index(p)
                    if ind == -1:
                        continue
                    if regions[n].points[ind-1] not in current_points and regions[n].points[ind-1] not in current_pnt:
                        pnts.append(regions[n].points[ind-1])
                        neighbors.append(n)
                    try:
                        if regions[n].points[ind+1] not in current_points and regions[n].points[ind+1] not in current_pnt:
                            pnts.append(regions[n].points[ind+1])
                            neighbors.append(n)
                    except:
                        pass
            if len(pnts) == 0:
                break
            pnt = []
            for p in pnts:
                if vec_x == 1 and vec_y ==0:
                    if n_ver[p][0] >= n_ver[river[-1]][0]:
                        pnt.append(p)
                if vec_x == 0 and vec_y ==1:
                    if n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == 0 and vec_y == -1:
                    if n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==0:
                    if n_ver[p][0] <= n_ver[river[-1]][0]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==1:
                    if n_ver[p][0] >= n_ver[river[-1]][0] and n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==-1:
                    if n_ver[p][0] <= n_ver[river[-1]][0] and n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == -1 and vec_y ==1:
                    if n_ver[p][0] <= n_ver[river[-1]][0] and n_ver[p][1] >= n_ver[river[-1]][1]:
                        pnt.append(p)
                if vec_x == 1 and vec_y ==-1:
                    if n_ver[p][0] >= n_ver[river[-1]][0] and n_ver[p][1] <= n_ver[river[-1]][1]:
                        pnt.append(p)

            if len(pnt) == 0:
                break
            p = rnd.choice(pnt)
            for n in neighbors:
                if p in regions[n].points:
                    neighbor = n
            if regions[neighbor].type == 0:
                break
            river.append(p)
            current_pnt.append(p)
            r = neighbor
        if len(river) > 2:
            current_points += current_pnt
            add_rivers.append(river)

    rivers += add_rivers

    to_json['regions'] = []
    for r in regions:
        to_json['regions'].append(r.to_json())


    print(time.time()-curtime, "seconds")
    print('Кількість суші:',len(lands)) 
    print('Кількість водного простору:',len(waters)) 
    print('Кількість річок:',len(rivers))
    to_json['lands'] = lands.tolist()
    to_json['waters'] = waters.tolist()
    to_json['rivers'] = rivers

    # with open('map_data.json', 'w') as f:
    #     json.dump(to_json, f, sort_keys=False, indent=1)

    img = Image.new('RGBA',(size_x,size_y), (0,0,0,255))    # Конечные границы
    idraw=ImageDraw.Draw(img)
    img2 = Image.new('RGBA',(size_x,size_y), (0,0,0,255))   # Границы без воды
    idraw2=ImageDraw.Draw(img2) 
    img3 = Image.new('RGBA',(size_x,size_y), (0,0,0,0))     # Просто суша и вода
    idraw3=ImageDraw.Draw(img3)
    img4 = Image.new('RGBA',(size_x,size_y), (0,0,0,0))     # Границы без цвета
    idraw4=ImageDraw.Draw(img4)
    img5 = Image.new('RGBA',(size_x,size_y), (0,0,0,255))   # Тип местности
    idraw5=ImageDraw.Draw(img5)
    img6 = Image.new('RGBA',(size_x,size_y), (0,0,0,0))     # Реки
    idraw6=ImageDraw.Draw(img6) 
    img7 = Image.new('RGBA',(size_x,size_y), (0,0,0,0))     # Картинки гор и деревьев
    idraw7=ImageDraw.Draw(img7)

    tree1= Image.open('stvol.png')
    tree2= Image.open('leaf.png')
    tree1 = tree1.resize((25,25))
    tree2 = tree2.resize((25,25))
    el = Image.open('el.png')
    el = el.resize((25,25))
    moun = Image.open('mountain.png')
    moun = moun.resize((40,40))

    print('Малюємо сушу.........')
    # Чертим сушу
    regions_ridges = []

    for r in lands:
        for n in regions[r].neighbors:
            if regions[n].type == 0:
                a = set(regions[r].points).intersection(regions[n].points)
                if len(a) >= 2:
                    a = list(a)
                    draw_line(n_ver[a[0]][0],  n_ver[a[0]][1], n_ver[a[1]][0], n_ver[a[1]][1], idraw3, (0,0,0, 255), 8, 0.7)
                    regions_ridges.append([n_ver[a[0]][0],  n_ver[a[0]][1], n_ver[a[1]][0], n_ver[a[1]][1]])
                    regions_ridges.append([n_ver[a[1]][0], n_ver[a[1]][1],n_ver[a[0]][0],  n_ver[a[0]][1]])
    img4.paste(img3)
    for lg in waters:
        ImageDraw.floodfill(img3, (new_points[lg][0], new_points[lg][1]), (0, 0 , 200 ,255))
    for lg in lands:
        ImageDraw.floodfill(img3, (new_points[lg][0], new_points[lg][1]), (40, 100 , 2 ,255))
        # idraw2.point(xy=((regions[lg].centr[0], regions[lg].centr[1])), fill = 'red')


    for lg in lands:
        if regions[lg].type == 1:
            idraw7.bitmap(((regions[lg].centr[0]-20,regions[lg].centr[1]-21)), moun, fill=(50,50,50,255))
        if regions[lg].type == 4:
            idraw7.bitmap(((regions[lg].centr[0]-11,regions[lg].centr[1]-11)), el, fill='green')
        if regions[lg].type == 7:
            idraw7.bitmap(((regions[lg].centr[0]-11,regions[lg].centr[1]-11)), tree2, fill='green')
            idraw7.bitmap(((regions[lg].centr[0]-11,regions[lg].centr[1]-11)), tree1, fill=(140,50,20,255))
            # idraw7.bitmap(((regions[lg].centr[0]-6,regions[lg].centr[1]-7)), tree2, fill='green')
            # idraw7.bitmap(((regions[lg].centr[0]-6,regions[lg].centr[1]-7)), tree1, fill=(140,50,20,255))
            # idraw7.bitmap(((regions[lg].centr[0]-16,regions[lg].centr[1]-7)), tree2, fill='green')
            # idraw7.bitmap(((regions[lg].centr[0]-16,regions[lg].centr[1]-7)), tree1, fill=(140,50,20,255))
        if regions[lg].type == 8:
            idraw7.bitmap(((regions[lg].centr[0]-11,regions[lg].centr[1]-11)), tree2, fill='green')
            idraw7.bitmap(((regions[lg].centr[0]-11,regions[lg].centr[1]-11)), tree1, fill=(140,50,20,255))

    print('Малюємо річки.........')
    # Рисуем реки
    for r in rivers:
        for p in range(len(r)-1):
            draw_line(n_ver[r[p]][0], n_ver[r[p]][1], n_ver[r[p+1]][0] , n_ver[r[p+1]][1], idraw6, (0,0,255, 255), 6,0.7)


    img2.paste(img3)
    print('Малюємо регіони.........')
    #Чертим кривые регионы
    for r in lands:
        for v in range(len(regions[r].points_cord)):
            temp = [regions[r].points_cord[v][0],  regions[r].points_cord[v][1], regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1]]
            tmp =  [regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1], regions[r].points_cord[v][0],  regions[r].points_cord[v][1]]
            if temp in regions_ridges or tmp in regions_ridges:
                continue
            regions_ridges.append(temp)
            regions_ridges.append(tmp)
            draw_line(regions[r].points_cord[v][0],  regions[r].points_cord[v][1], regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1], idraw4, (0,0,0, 55),6, 0.5)

    print(time.time()-curtime, "seconds")
    img2.alpha_composite(img6)
    img2.alpha_composite(img4)
    img2.alpha_composite(img7)
    font = ImageFont.truetype(font = 'arial.ttf',size = 8)
    # for lg in lands:
    #     ImageDraw.floodfill(img2, (new_points[lg][0], new_points[lg][1]), (40, 100 , 2 ,255))
    #     # idraw2.text(xy =(((new_points[lg][0], new_points[lg][1]))), text=str(lg), font=font,align='center', fill = (0,0,0,80))
    #     # idraw2.point(xy=((regions[lg].centr[0], regions[lg].centr[1])), fill = 'red')
    # for lg in waters:
    #     ImageDraw.floodfill(img2, (new_points[lg][0], new_points[lg][1]), (0, 0 , 200 ,255))

    img5.paste(img3)
    img3.alpha_composite(img6)
    img3.alpha_composite(img7)
    img5.alpha_composite(img4)

    print('Малюємо тип місцевості.........')
    for r in lands:
        # if r.type == 0:
        #     ImageDraw.floodfill(img5, (r.centr[0], r.centr[1]), (0, 0 , 200 ,255))
        if regions[r].type == 1:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (180, 60 , 35 ,255))
            idraw5.text(xy =(((regions[r].centr[0] - 4, regions[r].centr[1]- 4))), text='Г', font=font,align='center' ,fill = (0,0,0,80))
        elif regions[r].type == 2:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (140, 80 , 30 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Х', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 3:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (40, 100 , 2 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Рав', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 4:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (43, 113 , 76 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Тай', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 5:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (101, 98 , 62 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Тун', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 6:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (59, 79 , 49 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Бол', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 7:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (40, 130 , 40 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Лес', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 8:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (40, 190 , 40 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Рощ', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 9:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (150, 190 , 40 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='С', font=font,align='center',fill = (0,0,0,80))
        # elif regions[r].type == 10:
        #     ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (0, 110 , 25 ,255))
        #     idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='Джунг', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 11:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (205, 150 , 0 ,255))
            idraw5.text(xy =(((regions[r].centr[0]- 4, regions[r].centr[1]- 4))), text='ЗЗ', font=font,align='center',fill = (0,0,0,80))
        elif regions[r].type == 12:
            ImageDraw.floodfill(img5, (regions[r].centr[0], regions[r].centr[1]), (255, 255 , 2 ,255))
            idraw5.text(xy =(((regions[r].centr[0], regions[r].centr[1]))), text='Пус', font=font,align='center',fill = (0,0,0,80))


    #Чертим воду (по приколу)
    print('Малюємо воду.........')
    img.paste(img2)
    for r in waters:
        for v in range(len(regions[r].points_cord)):
            temp = [regions[r].points_cord[v][0],  regions[r].points_cord[v][1], regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1]]
            tmp =  [regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1], regions[r].points_cord[v][0],  regions[r].points_cord[v][1]]
            if temp in regions_ridges or tmp in regions_ridges:
                continue
            regions_ridges.append(temp)
            regions_ridges.append(tmp)
            idraw.line(xy=((regions[r].points_cord[v][0],  regions[r].points_cord[v][1]),(regions[r].points_cord[v-1][0], regions[r].points_cord[v-1][1])), fill=(0,0,0, 10), width=1) 

    # img4.save('Transperent.png')
    # img3.save('Map_WithoutRegions.png')
    # img2.save('Map_Borders.png')
    # img.save('Map_FullBorders.png')
    # img5.save('Terrain.png')
    # img6.save('Rivers.png')

    image_bytes = io.BytesIO()
    img.save(image_bytes, format='PNG')
    image_bytes2 = io.BytesIO()
    img2.save(image_bytes2, format='PNG')
    image_bytes3 = io.BytesIO()
    img3.save(image_bytes3, format='PNG')
    image_bytes4 = io.BytesIO()
    img4.save(image_bytes4, format='PNG')
    image_bytes5 = io.BytesIO()
    img5.save(image_bytes5, format='PNG')
    image_bytes6 = io.BytesIO()
    img6.save(image_bytes6, format='PNG')


    to_json['full_borders'] = image_bytes.getvalue()
    to_json['borders'] = image_bytes2.getvalue()
    to_json['without_regions'] = image_bytes3.getvalue()
    to_json['transperent'] = image_bytes4.getvalue()
    to_json['terrain'] = image_bytes5.getvalue()
    to_json['rivers'] = image_bytes6.getvalue()
    
    col = db_client['VictorGame']['maps']
    col.insert_one(to_json)


    print(time.time()-curtime, "seconds")
