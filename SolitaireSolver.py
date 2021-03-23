from PIL import ImageGrab, ImageFilter, Image, ImageDraw
import win32api
import win32con
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
import gc

sys.setrecursionlimit(1000)

# Rewrite from recursion to increment

# map?
# [[0],[[1],[2],[3],[4],[5],[6],[7],[8],[9]]]
# move
# [from there, from_deep, to pos n, auto? = true\false]
# pos
#
# possible cards: 4*(6..10) + 4*4
# how to store this?
# for numbers only red and black matters (red, black) = (1, 2)
# (6 .. 10) = (6 .. 10)
# for pictures (Jack, Queen, King, Ace) = -1
#              (Hearts, Diamonds, Spades, Clubs) = (3, 4, 5, 6)
# 1,6  2,6
# 1,7  2,7
# 1,8  2,8
# 1,9  2,9
# 1,10 2,10
# 3,-1  4,-1  5,-1  6,-1
# -1,-1 (card back)


class screen_positions:
    def __init__(self):
        self.free_cell = ()                 # x, y, w, h
        self.main_field = ()                # x, y, w, h
        self.main = []                      # x1, x2, x3, x4, x5, x6, x7, x8
        self.card = ()                      # w, h

cards_pos = screen_positions()
cards_pos.free_cell = (1365, 214, 125, 187)
cards_pos.main_field = (362, 457, 1434+125, 457+277)
cards_pos.main = [1434, 1300, 1166, 1032, 898, 764, 630, 496, 362]
cards_pos.card = (125, 187)
cards_pos.main.sort()

# (1434, 457, 125, 277)
# (1300, 457, 125, 277)
# (1166, 457, 125, 277)
# (1032, 457, 125, 277)
# (898, 457, 125, 277)
# (764, 457, 125, 277)
# (630, 457, 125, 277)
# (496, 457, 125, 277)
# (362, 457, 125, 277)

data = {}
move_weights = [10, 4, 1, 0]
# full_chain = 10,
# full_chain_to_free = 4, to_free_cell = 1, break_chain = 1 ...


class Move:
    # map_pos (0,((1),(2),(3),(4),(5),(6),(7),(8),(9))
    def __init__(self, from_pos, deep, to, weight):
        self.from_pos = from_pos
        # deep calculated from bottom to top, so last card at 0 deep
        self.deep = deep
        self.to = to
        self.weight = weight

    def __repr__(self):
        return "M f:%s d:%s to:%s w:%s" % (self.from_pos, self.deep, self.to, self.weight)

    def __str__(self):
        str_ = "Move from position %s with deep of %s to position %s, weight=%s"
        return str_ % (self.from_pos, self.deep, self.to, self.weight)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (self.__dict__ == other.__dict__)

    def __le__(self, other):
        return self.weight <= other.weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __ge__(self, other):
        return self.weight >= other.weight


def card_str(card):
    return "(%s, %2s)" % card if card != () else "(     )"


class Map_:
    # Solitiare map
    def __init__(self):
        self.free_cell = ()
        self.main = [[], [], [], [], [], [], [], [], []]

    def __str__(self):  # map representation
        top = " "*60 + card_str(self.free_cell) + "\n"
        b_n = max([len(k) for k in self.main] + [1])
        b = [k + [()]*(b_n-len(k)) for k in self.main]
        main = top + "\n".join(" ".join(card_str(k[i]) for k in b) for i in range(len(b[0])))
        return main

    def __eq__(self, other):
        return isinstance(other, self.__class__) and (hash(self) == hash(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(tuple(c) for c in sorted(self.main)))

    def copy_map(self, map):
        self.main = [list(a) for a in map.main]

    def n(self):
        return (self.free_cell != ()) + sum(sum(k != () for k in a) for a in self.main)

    def read_map(self):
        # read screen and evaluate map from it
        # 1. find basic contour
        # 2. adjust coorinates, make coordinate map
        # 2.5 make database from chips
        # 3. compare imgs with base to read value
        global cards_pos
        global data

        data_map = {}
        printProgressBar('Reading screen:', 30, 0)
        screenshot = ImageGrab.grab()  # Make screenshot
        printProgressBar('Reading screen:', 30, 0.3)
        img_rgb = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
        printProgressBar('Reading screen:', 30, 0.6)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        printProgressBar('Reading screen:', 30, 1)
        print()

        x_delta = 0
        y_delta = 30
        x_diff = 6
        y_diff = 5
        x_size = 27
        y_size = 22

        bar_max, bar_i = len(data), 0    # progress bar
        printProgressBar('Building map:', 30, bar_i/bar_max)
        for key, template in data.items():  # find all mathces with database
            a = []
            w, h = template.shape[::-1]
            template_color = cv2.mean(template)
            #print(key, template_color)
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            if key[1] > 0:
                threshold = 0.99
            else:
                threshold = 0.95
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                #print(pt)
                color = cv2.mean(img_gray[pt[1]:pt[1] + h, pt[0]:pt[0]+w])
                if not abs(color[0] - template_color[0]) < 5:  # Check color match
                    continue
                x, y = pt
                y_c = cards_pos.main_field[1]
                # Add check for free cell pos
                # TODO: fix it, can't read it
                if abs(x - cards_pos.free_cell[0]) < 20 and abs(y - cards_pos.free_cell[1]) < 20:
                    self.free_cell = key
                if y >= y_c - 20:
                    for n, x_c in enumerate(cards_pos.main):  # check main field
                        if abs(x_c + x_diff - x) < 20:
                            i = int((y - y_diff - y_c)/y_delta)
                            if len(self.main[n]) < i + 1:
                                self.main[n] += [()]*(i - len(self.main[n]) + 1)
                            self.main[n][i] = key
                a.append(pt)
                cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                #cv2.imshow('image', img_gray)
                #cv2.waitKey(0)
            data_map[key] = a  # create coordinates matches database
            #print(self)
            bar_i += 1
            printProgressBar('Building map:', 30, bar_i/bar_max)
        print()
        #cv2.imshow('image', img_gray)
        #cv2.waitKey(0)
        # func end

    def possible_moves(self):
        # list of all possible moves from given map
        move_list = []

        if self.free_cell == (): # free_cell is empty
            move_list += [Move(i+1, 0, 0, 1) for i, k in enumerate(self.main) if k != []]

        for i, column in enumerate(self.main):  # check main field column by column
            if column != []:
                max_valid_chain_len = find_max_valid_chain_len(column)
                for d, card in enumerate(column):  # c = card, d = card num in column
                    if len(column) - d == max_valid_chain_len:  # max chain
                        w = 10
                    else:
                        w = 1
                    # if d + 1 == len(x): # only last card
                    #    move_list += [Move(i+1, 0, j+1, w) for j,y in enumerate(self.main) if j!=i and (y == [] or check_valid_junction(x[d], y[-1]))]
                    # elif  chek_valid_chain(x[d:]): # valid chain of cards
                    if d >= len(column) - max_valid_chain_len:
                        for j, to_column in enumerate(self.main):
                            if j != i and (to_column == [] or check_valid_junction(column[d], to_column[-1])):
                                if to_column == []:
                                    w = w % 6
                                move_list.append(Move(i+1, len(column)-d-1, j+1, w))
        return move_list

    # map_pos (0,((1),(2),(3),(4),(5),(6),(7),(8),(9)))

    def move_execute(self, m):
        # execute move on current map
        # new_map = Map_()
        # new_map.copy(self)
        # map_pos (0,((1),(2),(3),(4),(5),(6),(7),(8),(9)))
        if m.from_pos == 0:
            a = self.free_cell
            self.free_cell = ()
        else:
            a = self.main[m.from_pos-1][-1-m.deep:]
            self.main[m.from_pos-1] = self.main[m.from_pos-1][:-1-m.deep]

        if m.to == 0:
            if type(a) == list:
                a = a[0]
            self.free_cell = a
        else:
            if type(a) is list:
                self.main[m.to-1] += a
            else:
                self.main[m.to-1].append(a)
        # return(new_map)
        # func end

win_map = Map_()
win_map.free_cell = ()
win_map.main = [
    [(1, 10), (2, 9), (1, 8), (2, 7), (1, 6)],
    [(1, 10), (2, 9), (1, 8), (2, 7), (1, 6)],
    [(2, 10), (1, 9), (2, 8), (1, 7), (2, 6)],
    [(2, 10), (1, 9), (2, 8), (1, 7), (2, 6)],
    [(3, -1), (3, -1), (3, -1), (3, -1)],
    [(4, -1), (4, -1), (4, -1), (4, -1)],
    [(5, -1), (5, -1), (5, -1), (5, -1)],
    [(6, -1), (6, -1), (6, -1), (6, -1)],
    []
]

test_map = Map_()
test_map.main = [
    [(6, -1), (5, -1), (4, -1), (2, 7)],
    [(1, 6), (4, -1), (3, -1), (5, -1)],
    [(2, 8), (1, 7), (5, -1), (2, 9)],
    [(2, 10), (2, 9), (6, -1), (4, -1)],
    [(1, 10), (4, -1), (2, 6), (3, -1)],
    [(1, 7), (6, -1), (3, -1), (1, 8)],
    [(5, -1), (2, 8), (3, -1), (6, -1)],
    [(2, 10), (1, 9), (2, 7), (1, 8)],
    [(2, 6), (1, 6), (1, 9), (1, 10)]
]


def printProgressBar(name, width, percent, symbol='#'):
    print('\r{0: <30} [{1:30s}] {2:.1f}%'.format(name, symbol * int(width * percent), percent * 100), end='')


def chek_valid_chain(chain):
    return all([check_valid_junction(b, a) for a, b in zip(chain, chain[1:])])


def find_max_valid_chain_len(chain):
    res = 1
    while len(chain) > res and check_valid_junction(chain[-res], chain[-res-1]):
        res += 1
    return res


def check_valid_junction(card, card_to):
    if card[0] != card_to[0] and card[1] + 1 == card_to[1]:
        return True
    if card[0] == card_to[0] and card[1] == -1:
        return True
    return False


def find_solution_path():
    # recursive creating solution path
    # each iteration check all possible moves and check if it lead us to victory
    # if dead end - drop that branch
    # if stuck in returning to prev position - drop
    global solution_found
    global map_dict      # dict={map: path_to_that_map, ...}
    global map_stack     # dict={move_weight: [maps...], } stack what map to check for possible moves next
    global map_checked   # set of maps that already checked (don't check it again)
    global move_weights  # list of possible move weights
    global win_map
    global step
    global all_steps
    min_n = 40
    while not solution_found:
        i = 0
        while map_stack[move_weights[i]] == []:
            i += 1
        w_ = move_weights[i]
        # w_ = max([w for w in move_weights if map_stack[w] != []] + [0])
        if w_ != 0:
            next_map = map_stack[w_].pop()
            step = step + 1
            if len(map_dict[next_map]) > MAX_PATH:
                continue
            if step > MAX_STEPS:
                print("\nReached MAX_STEPS count, skipping this board.")
                return False
            # if next_map in map_checked:
            #    print("lalalalala")
            #    continue # why no entry???
            # else:
            #    map_checked.add(next_map)
            prev_path = map_dict[next_map]
            moves_to_test = next_map.possible_moves()
            #print(moves_to_test)
            #exit()
            # moves_to_test.sort(reverse = True)
            for test_move in moves_to_test:
                new_map = Map_()
                new_map.copy_map(next_map)
                new_map.move_execute(test_move)

                if map_dict.get(new_map) is None:
                    map_dict[new_map] = prev_path + [test_move]
                    map_stack[test_move.weight].append(new_map)
                else:
                    if len(map_dict[new_map]) > len(prev_path) + 1:
                        map_dict[new_map] = prev_path + [test_move]
                #if new_map.n() < min_n:
                #    min_n = new_map.n()
                if new_map == win_map:
                    solution_found = True
                    break
            all_steps = all_steps + len(moves_to_test)
            percent = (40 - min_n)/40  # draw progressbar
            str_ = '\r{0: <30} [{1:30s}] {2:.1f}%    {3:d} / {4:d}  moves checked'
            print(str_.format('Finding solution:', '#'*int(30*percent), percent*100, step, all_steps), end='')
        else:
            break
    # print()


def solution_optimization():
    # optimize given solution
    global map_dict
    global win_map
    move_list = map_dict[win_map]
    # how optimize it?
    # - figure out which one moves are useless, but how?
    # - determine move weight depending on how usefull it
    # like this:
    #   move to topright is 1
    #   move to freecell is 5
    #   move to another card is 10
    #   spec button move is 50
    #   auto-move is 25
    #   ....
    #   depend of this check most value moves first
    #   amd all of it should be done in find_solution_path()
    #   oh well...
    pass


def solve_it(sc_map, path):
    # map_pos ((0,1,2),3,(4,5,6),((7),(9),(9),(10),(11),(12),(13),(14)))
    # map_pos (0,((1),(2),(3),(4),(5),(6),(7),(8),(9))
    _, _, card_w, card_h = cards_pos.free_cell
    card_w = card_w // 2
    card_h = card_h // 2
    y_delta = 31
    click(100, 100)
    time.sleep(0.5)
    for m in path:
        print(m)
        x, y = 0, 0
        if m.from_pos == 0:
            x, y, _, _ = cards_pos.free_cell
            x += card_w
            y += card_h
        else:
            x = cards_pos.main[m.from_pos-1] + card_w
            y = cards_pos.main_field[1] + 10 + y_delta*(len(sc_map.main[m.from_pos-1])-1-m.deep)

        x2, y2 = 0, 0
        if m.to == 0:
            x2, y2, _, _ = cards_pos.free_cell
            x2 += card_w
            y2 += card_h
        else:
            x2 = cards_pos.main[m.to-1] + card_w
            i = len(sc_map.main[m.to-1])
            i = i if i != 0 else 1
            y2 = cards_pos.main_field[1] + 10 + y_delta*(i-1)
        # time.sleep(0.3)
        # screenshot =  ImageGrab.grab()
        # draw = ImageDraw.Draw(screenshot)
        # draw.line((x,y, x2,y2), fill=128)
        # img_rgb = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
        # cv2.imshow('image',img_rgb)
        # cv2.moveWindow('image', 1366, -350)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # click(100, 100)
        time.sleep(0.3)
        if x2 != 0:
            move_click(x, y, x2, y2)
        else:
            click(x, y)
        sc_map.move_execute(m)
        # break
    # solve it by move mouse and so


def find_contours():
    screenshot = ImageGrab.grab()  # Make screenshot
    im = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Show", thresh)
    # cv2.waitKey(0)
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt=contours[max_index]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and w < 200 and h > 180:
            # print("x = %4s   y = %4s   h = %4s   w = %4s"%(x, y, w, h))
            print((x, y, w, h))
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Show", im)
    cv2.waitKey(0)


def make_database():
    _, y, _, _ = cards_pos.main_field
    w, h = cards_pos.card
    screenshot = ImageGrab.grab()  # Make screenshot
    img_rgb = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
    fig = plt.figure()
    x_delta = 0
    y_delta = 30
    x_diff = 6
    y_diff = 5
    x_size = 27
    y_size = 22
    n = 0
    for j, x in enumerate(cards_pos.main):
        for i in range(4):
            # cv2.imshow('image',img_rgb[y+y_diff:y+y_diff+y_size, x+x_diff:x+x_diff+x_size])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            ax = fig.add_subplot(4, 9, i*9 + 1 + j)
            img_fragment = img_rgb[y+y_diff+y_delta*i:y+y_diff+y_delta*i+y_size, x+x_diff+x_delta*i:x+x_diff+x_delta*i+x_size]
            ax.imshow(img_fragment, 'gray')
            cv2.imwrite(str(n) + ".png", img_fragment)
            n += 1
            ax.autoscale(False)
            ax.axis('off')
    plt.show()


def read_database():
    global data
    imgs = glob.glob("data\*.png")
    bar_max, bar_i = len(imgs), 0  # progress bar
    for a in imgs:
        data[tuple(int(x) for x in a[a.index('(')+1:a.index(')')].split(','))] = cv2.imread(a, 0)
        bar_i = bar_i + 1  # draw progressbar
        printProgressBar('Reading img database:', 30, bar_i/bar_max)
    print()


def click(x, y):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def move_click(x, y, x2, y2):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    time.sleep(0.1)
    win32api.SetCursorPos((x2, y2))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x2, y2, 0, 0)

MAX_PATH = 150
AUTO_SOLVE = True
MAX_STEPS = 100000

if __name__ == "__main__":

    print("Program start.")
    #find_contours()
    #make_database()
    #exit()
    read_database()
    while AUTO_SOLVE is True:
        step = 0
        all_steps = 0
        solution_found = False
        map_dict = {}
        map_stack = {}
        map_checked = set()
        gc.collect()
        screen_map = Map_()
        screen_map.read_map()
        #print(screen_map.main)
        #screen_map = test_map
        for w in move_weights:
            map_stack[w] = []
        map_stack[0].append(None)
        map_stack[10].append(screen_map)
        map_dict[screen_map] = []

        print("\nCurrent Map:\n")
        print(screen_map)
        print("\n")
        #exit()
        # click(50, 50)
        find_better_solution = True
        find_solution_path()
        if solution_found:
            while find_better_solution:
                all_moves, auto_moves = len(map_dict[win_map]), len([1 for m in map_dict[win_map] if m.weight == 10])
                print("\nSolution found: %s moves (%s manual, %s auto)" % (all_moves, all_moves - auto_moves, auto_moves))
                if not AUTO_SOLVE:
                    print("Find better solution? (y/n): ", end='')
                    ans = input()
                    if ans == 'y':
                        solution_found = False
                        while len(map_dict[win_map]) == all_moves:
                            try:
                                find_solution_path()
                            except KeyboardInterrupt:
                                print()
                                print("Keyboard Interrupt detected, stop better solution search")
                                break
                            # print("Another solution found: same moves number")
                            solution_found = False
                        if len(map_stack) == 0:
                            print("No possible moves left")
                            find_better_solution = False
                    else:
                        find_better_solution = False
                else:
                    find_better_solution = False
            if solution_found:
                solve_it(screen_map, map_dict[win_map])
                time.sleep(3)
        else:
            print("Reach dead end, no solution detected.")
        click(1467, 900)
        time.sleep(6)

    print("Program end.")
