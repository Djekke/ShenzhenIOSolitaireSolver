from PIL import ImageGrab, ImageFilter, Image, ImageDraw
import win32api, win32con
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
import gc

sys.setrecursionlimit(1000)

# Rewrite from recursion to increment

## map?
## [[0,0,0],1,[2,3,4],[[5],[6],[7],[8],[9],[10],[11],[12]))
## move
## [from there, from_deep, to pos n, auto? = true\false]
## pos
## spec-button move? (0, 0, 0, False)
## 
## possible cards: 3*(1..9) + 3*4 + 1
## how to store this?
## like this: (rgb = 123) (red green black :D )
## 1,1 2,1 3,1
## 1,2 2,2 3,2
## 1,3 2,3 3,3
## ...
## 1,9 2,9 3,9
## 4,-1 5,-1 6,-1 (spec cards x4)
## 7,-1 (special middle card)
## 8,-1 (card back)

class screen_positions:
	def __init__(self):
		self.spec_buttons = [(), (), ()]	# x, y, w, h
		self.cells = [(), (), ()]			# x, y, w, h
		self.middle = ()					# x, y, w, h
		self.topright = [(), (), ()]		# x, y, w, h
		self.main_field = ()				# x, y, w, h
		self.main = []						# x1, x2, x3, x4, x5, x6, x7, x8

cards_pos = screen_positions()
cards_pos.spec_buttons = [ 
(575, 25, 53, 54),
(575, 108, 53, 54),
(575, 191, 53, 54)
]
cards_pos.cells = [
(116, 19, 120, 232),
(268, 19, 120, 232),
(420, 19, 120, 232)
]
cards_pos.middle = (684, 19, 120, 232)
cards_pos.topright = [
(876, 19, 120, 232),
(1028, 16, 120, 233),
(1180, 18, 120, 232)
]
cards_pos.main_field = (116, 283, 1180+121, 283+418)
cards_pos.main = [1180, 1028, 876, 724, 572, 420, 268, 116]
cards_pos.main.sort()

data = {}
move_weights = [50, 40, 10, 4, 1, 0]
		# auto_move = 50, special_move = 40, full_chain = 10, 
		# full_chain_to_free = 4, to_free_cell = 1, break_chain = 1 ...
class Move:
	# map_pos ((0,1,2),3,(4,5,6),((7),(9),(9),(10),(11),(12),(13),(14)))
	def __init__(self, from_pos, deep, to, weight):
		self.from_pos = from_pos
		self.deep = deep #deep calculated from bottom to top, so last card at 0 deep
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
	return "(%s, %2s)"%card if card != () else "(     )" 

class Map_:
	# Solitiare map
	def __init__(self):
		self.cells = [ (), (), () ]
		self.middle = False
		self.topright = [ (), (), () ]
		self.main = [ [], [], [], [], [], [], [], [] ]
	
	def __str__(self): #map representation
		t1 = " ".join(card_str(k) for k in self.cells)
		t2 = "(7, -1)" if self.middle else "(     )"
		t3 = " ".join(card_str(k) for k in self.topright)
		top = t1 + "     " + t2 + "     " + t3
		b_n = max([len(k) for k in self.main] + [1])
		b = [k + [()]*(b_n-len(k)) for k in self.main]
		main = "\n".join(" ".join(card_str(k[i]) for k in b) for i in range(len(b[0])))
		return top + "\n\n" + main
	
	def __eq__(self, other):
		#return( self.__dict__ == other.__dict__ )
		return isinstance(other, self.__class__) and (hash(self) == hash(other)) 
	
	def __ne__(self, other):
		return not self.__eq__(other) 
	
	def __hash__(self):
		#return(hash(str(self)))
		return hash((tuple(sorted(self.cells)), self.middle, tuple(sorted(self.topright)), tuple(tuple(c) for c in sorted(self.main))))
	
	def copy_map(self, map):
		self.cells = list(map.cells)
		self.middle = map.middle
		self.topright = list(map.topright)
		self.main = [list(a) for a in map.main]
	
	def n(self):
		return sum(k != () and k != (8, -1) for k in self.cells) + sum(sum(k != () for k in a) for a in self.main)
	
	def read_map(self):
		#read screen and evaluate map from it
		
		# 1. find basic contour
		# 2. adjust coorinates, make coordinate map
		# 2.5 make database from chips
		# 3. compare imgs with base to read value
		global cards_pos
		global data
		
		data_map = {}
		printProgressBar('Reading screen:', 30, 0)
		screenshot =  ImageGrab.grab() # Make screenshot
		printProgressBar('Reading screen:', 30, 0.3)
		img_rgb = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
		printProgressBar('Reading screen:', 30, 0.6)
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
		printProgressBar('Reading screen:', 30, 1)
		print()
		
		y_delta = 31
		x_delta = 0
		y_diff = 8
		x_diff = 8
		x_size = 20
		y_size = 23
		
		bar_max, bar_i = len(data), 0	#progress bar
		for key, value in data.items(): #find all mathces with database
			a = []
			#print(key)
			template = value
			w, h = template.shape[::-1]
			res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
			threshold = 0.99
			loc = np.where(res >= threshold)
			for pt in zip(*loc[::-1]):
				x, y = pt
				for n, coord in enumerate(cards_pos.cells): #check cells pos
					x_c, y_c, _, _ = coord
					if abs(x_c + x_diff - x) < 20 and abs(y_c + y_diff - y ) < 20:
						self.cells[n] = key
				if key == (7, -1) and abs(cards_pos.middle[0] - x) < 20 and abs(cards_pos.middle[1] - y) < 20 :
					self.middle = True #check middle
				for n, coord in enumerate(cards_pos.topright): #check topright pos
					x_c, y_c, _, _ = coord
					if abs(x_c + x_diff - x) < 20 and abs(y_c + y_diff - y ) < 20:
						self.topright[n] = key
				y_c = cards_pos.main_field[1]
				if y >= y_c - 20:
					for n, x_c in enumerate(cards_pos.main): #check main field
						if abs(x_c + x_diff - x) < 20:
							i = int((y - y_diff - y_c)/y_delta)
							if len(self.main[n]) < i + 1:
								self.main[n] += [()]*(i - len(self.main[n]) + 1)
							self.main[n][i] = key
				a.append(pt)
				#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			data_map[key] = a #create coordinates matches database
			bar_i += 1
			printProgressBar('Building map:', 30, bar_i/bar_max)
		print()
		#func end
	
	def possible_moves(self):
		#list of all possible moves from given map
		move_list = []
		
		if () in self.cells: #any top-left cell is empty
			j = self.cells.index(()) # index of free cell
			move_list += [Move(i+7, 0, j, 1) for i, k in enumerate(self.main) if k != []]
			# do not check moves from topright to freecells, becouse it's stupid move !!!
		
		for i, x in enumerate(self.cells):	#any top-left cells is ocupy, check if we can drop it down to main field
			if x != () and x != (8, -1): #valid card on that spot
				#check topright cells
				# ... "or to topright" move that part to automove
				#move_list += [Move(i, 0, j+4, False) for j,y in enumerate(self.topright) if y and check_valid_junction(x, y)]
				#check main field
				move_list += [Move(i, 0, j+7, 1 if y == [] else 10) for j, y in enumerate(self.main) if y == [] or check_valid_junction(x, y[-1])]
		
		#comment this section to check this kind of moves is needed at all
		"""
		if any(self.topright): #check if we can move from topright to mainfield (sometimes it usefull, yes?)
			for i,x in enumerate(self.topright):
				if x:
					move_list += [Move(i+4, 0, j+7, False) for j,y in enumerate(self.main) if (y and check_valid_junction(x, y[-1])) or not y]
		"""
		#...
		for i, x in enumerate(self.main): #check main field column by column
			if x != []: # x = column
				n = find_max_valid_chain_len(x)
				for d, c in enumerate(x): # c = card, d = card num in column
					if len(x)-d == n: # max chain
						w = 10
					else:
						w = 1
					#if d + 1 == len(x): # only last card
					#	move_list += [Move(i+7, 0, j+7, w) for j,y in enumerate(self.main) if j!=i and (y == [] or check_valid_junction(x[d], y[-1]))]
					#elif  chek_valid_chain(x[d:]): # valid chain of cards
					if d >= len(x)-n:
						for j, y in enumerate(self.main):
							if j!=i and (y == [] or check_valid_junction(x[d], y[-1])):
								if y == []:
									w = w % 6
								move_list.append(Move(i+7, len(x)-d-1, j+7, w))
		
		#check for special buttons move (if all 4 of same type special card available and we have a free cell)
		# special cards is  (4,-1) (5,-1) (6,-1) (spec cards x4)
		for sp in [(4,-1), (5,-1), (6,-1)]:
			if self.cells.count(sp) + [x[-1] for x in self.main if x != []].count(sp) == 4: #all 4 of 1 type available to move
				if sp in self.cells or () in self.cells:
					move_list.append(Move(15, sp[0], 15, 40))
		
		return move_list
	
	# map_pos ((0,1,2),3,(4,5,6),((7),(9),(9),(10),(11),(12),(13),(14)))
	def auto_move(self):
		#check for automatic move, that game do for me 
		#return auto move or Null if nothing to do
		
		for i, k in enumerate(self.main):
			if (7, -1) in k:
				if k[-1] == (7,-1):
					return Move(i+7, 0, 3, 50)
		
		min_n = min([a[1] if a != () else 0 for a in self.topright])
		
		for i, k in enumerate(self.cells):
			for x in [1, 2, 3]:
				if k == (x, min_n+1):
					if min_n == 0:
						j = self.topright.index(())
					else:
						j = self.topright.index((x, min_n))
					return Move(i, 0, j+4, 50)
		
		for i, k in enumerate(self.main): # because of this automove can stuck it all in loop
			if k != []:	# if not empty column in self.main
				for x in [1, 2, 3]:
					if k[-1] == (x, min_n+1):
						if min_n == 0:
							j = self.topright.index(())
						else:
							j = self.topright.index((x, min_n))
						return Move(i+7, 0, j+4, 50)
		return False
	
	def move_execute(self, m):
		# execute move on current map
		#new_map = Map_()
		#new_map.copy(self)
		# map_pos ((0,1,2),3,(4,5,6),((7),(9),(9),(10),(11),(12),(13),(14)))
		if m.from_pos < 3:
			a = self.cells[m.from_pos]
			self.cells[m.from_pos] = ()
		elif m.from_pos < 4:
			pass
		elif m.from_pos < 7:
			a = self.topright[m.from_pos-4]
			if a[1] > 1:
				self.topright[m.from_pos-4][1] = a[1] - 1
			else: #don't know is it possible to remove card N 1
				self.topright[m.from_pos-4] = ()
		elif m.from_pos < 15:
			a = self.main[m.from_pos-7][-1-m.deep:]
			self.main[m.from_pos-7] = self.main[m.from_pos-7][:-1-m.deep]
		else: #special button moves
			a = (m.deep, -1)
			for k in self.main:
				if a in k:
					k.pop()
			for i, k in enumerate(self.cells):
				if a == k:
					self.cells[i] = ()
			self.cells[self.cells.index(())] = (8, -1)
		
		if m.to < 3:
			if type(a) == list:
				a = a[0]
			self.cells[m.to] = a
		elif m.to < 4:
			self.middle = True
		elif m.to < 7:
			if type(a) == list:
				if a == []:
					raise WtfError
				a = a[0]
			self.topright[m.to-4] = a
		elif m.to < 15:
			if type(a) is list:
				self.main[m.to-7] += a
			else:
				self.main[m.to-7].append(a)
		#return(new_map)
		#func end



win_map = Map_()
win_map.cells = [(8, -1), (8, -1), (8, -1)]
win_map.middle = True
win_map.topright = [(1, 9), (2, 9), (3, 9)]
win_map.main = [ [], [], [], [], [], [], [], [] ]

test_map = Map_()
test_map.main = [
 [ (2, 8), (3, 6), (5,-1), (7,-1), (3, 8) ],
 [ (4,-1), (3, 3), (5,-1), (2, 7), (2, 2) ],
 [ (3, 4), (1, 3), (2, 4), (2, 9), (4,-1) ],
 [ (6,-1), (3, 1), (6,-1), (1, 6), (2, 6) ],
 [ (5,-1), (2, 3), (3, 9), (6,-1), (3, 5) ],
 [ (1, 9), (3, 7), (4,-1), (2, 5), (3, 2) ], 
 [ (1, 5), (1, 2), (5,-1), (1, 4), (6,-1) ], 
 [ (2, 1), (1, 8), (1, 1), (4,-1), (1, 7) ], 
]
test_map2 = Map_()
test_map2.main = [
 [ (2, 8), (3, 6), (5,-1), (7,-1), (3, 8) ],
 [ (4,-1), (3, 3), (5,-1), (2, 7), (2, 2) ],
 [ (3, 4), (1, 3), (2, 4), (2, 9), (4,-1) ],
 [ (6,-1), (3, 1), (6,-1), (1, 6), (2, 6) ],
 [ (5,-1), (2, 3), (3, 9), (6,-1), (3, 5) ],
 [ (1, 9), (3, 7), (4,-1), (2, 5), (3, 2) ], 
 [ (1, 5), (1, 2), (5,-1), (1, 4), (6,-1) ], 
 [ (2, 1), (1, 8), (1, 1), (4,-1), (1, 7) ], 
]
 
def printProgressBar(name, width, percent, symbol = '#'):
	print('\r{0: <30} [{1:30s}] {2:.1f}%'.format(name, symbol * int(width * percent), percent * 100), end='')

def chek_valid_chain(chain):
	return all([check_valid_junction(b,a) for a,b in zip(chain, chain[1:])])

def find_max_valid_chain_len(chain):
	res = 1
	while len(chain) > res and check_valid_junction(chain[-res], chain[-res-1]):
		res += 1
	return res

def check_valid_junction(card, card_to):
	if card[0] != card_to[0] and card[1] + 1 == card_to[1]:
		return True
	return False

def find_solution_path():
	#recursive creating solution path
	#each iteration check all possible moves and check if it lead us to victory
	#if dead end - drop that branch
	#if stuck in returning to prev position - drop
	#chech for auto-move befor each move
	global solution_found
	global map_dict		# dict={map: path_to_that_map, ...}
	global map_stack	# dict={move_weight: [maps...], } stuck what map to check on possible moves next
	global map_checked	# set of maps that already checked (not check it again)
	global move_weights # list of possible move weights
	global win_map
	global step
	global all_steps
	min_n = 40
	while not solution_found:
		i = 0
		while map_stack[move_weights[i]] == []: i += 1
		w_ = move_weights[i]
		#w_ = max([w for w in move_weights if map_stack[w] != []] + [0])
		if w_ != 0:
			next_map = map_stack[w_].pop()
			step = step + 1
			if len(map_dict[next_map]) > MAX_PATH:
				continue
			if step > MAX_STEPS:
				print("\nReached MAX_STEPS count, skipping this board.")
				return False
			#if next_map in map_checked:
			#	print("lalalalala")
			#	continue # why no entry???
			#else:
			#	map_checked.add(next_map)
			prev_path = map_dict[next_map]
			a = next_map.auto_move()
			if a:
				moves_to_test = [a]
			else:
				moves_to_test = next_map.possible_moves()
			#moves_to_test.sort(reverse = True)
			for test_move in moves_to_test:
				new_map = Map_()
				new_map.copy_map(next_map)
				new_map.move_execute(test_move)
				
				if map_dict.get(new_map) == None:
					map_dict[new_map] = prev_path + [test_move]
					map_stack[test_move.weight].append(new_map)
				else:
					if len(map_dict[new_map]) > len(prev_path) + 1:
						map_dict[new_map] = prev_path + [test_move]
				if new_map.n() < min_n:
					min_n = new_map.n()
				if new_map == win_map:
					solution_found = True
					break
			all_steps = all_steps + len(moves_to_test)
			percent = (40 - min_n)/40	#draw progressbar
			str_ = '\r{0: <30} [{1:30s}] {2:.1f}%    {3:d} / {4:d}  moves checked'
			print(str_.format('Finding solution:', '#'*int(30*percent), percent*100, step, all_steps), end='')
		else:
			break
	#print()

def solution_optimization():
	#optimize given solution
	global map_dict
	global win_map
	move_list = map_dict[win_map]
	# how optimize it?
	# - figure out which one moves are useless, but how?
	# - determine move weight depending on how usefull it
	# like this:
	#	move to topright is 1
	#	move to freecell is 5
	#	move to another card is 10
	#	spec button move is 50
	#	auto-move is 25
	#	....
	#	depend of this check most value moves first
	#	amd all of it should be done in find_solution_path()
	#	oh well...
	pass

def solve_it(sc_map, path):
	# map_pos ((0,1,2),3,(4,5,6),((7),(9),(9),(10),(11),(12),(13),(14)))
	_, _ ,card_w, card_h = cards_pos.cells[0]
	card_w = card_w // 2
	card_h = card_h // 2
	y_delta = 31
	click(100, 100)
	time.sleep(0.5)
	for m in path:
		print(m)
		if m.weight != 50:
			x, y = 0, 0
			if m.from_pos < 3:
				x, y, _, _ = cards_pos.cells[m.from_pos]
				x += card_w
				y += card_h
			elif m.from_pos < 7:
				#a = self.topright[m.from_pos-4]
				#if a[1] > 1:
				#	self.topright[m.from_pos-4][1] = a[1] - 1
				#else: #don't know is it possible to remove card N 1
				#	self.topright[m.from_pos-4] = ()
				print("nothing here")
			elif m.from_pos < 15:
				x = cards_pos.main[m.from_pos-7] + card_w
				y = cards_pos.main_field[1] + 10 + y_delta*(len(sc_map.main[m.from_pos-7])-1-m.deep)
			else: #special button moves
				x, y, w, h = cards_pos.spec_buttons[m.deep-4]
				x += w // 2
				y += h // 2
			x2, y2 = 0, 0
			if m.to < 3:
				x2, y2, _, _ = cards_pos.cells[m.to]
				x2 += card_w
				y2 += card_h
			elif m.to < 4:
				pass
			elif m.to < 7:
				x2, y2, _, _ = cards_pos.topright[m.to-4]
				x2 += card_w
				y2 += card_h
			elif m.to < 15:
				x2 = cards_pos.main[m.to-7] + card_w
				i = len(sc_map.main[m.to-7])
				i = i if i != 0 else 1
				y2 = cards_pos.main_field[1] + 10 + y_delta*(i-1)
			#time.sleep(0.3)
			#screenshot =  ImageGrab.grab()
			#draw = ImageDraw.Draw(screenshot) 
			#draw.line((x,y, x2,y2), fill=128)
			#img_rgb = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
			#cv2.imshow('image',img_rgb)
			#cv2.moveWindow('image', 1366, -350)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			#click(100, 100)
			time.sleep(0.3)
			if x2 != 0:
				move_click(x, y, x2, y2)
			else:
				click(x, y)
		else:
			time.sleep(0.4)
		sc_map.move_execute(m)
		#break
	#solve it by move mouse and so

def find_contours(im):
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,0)
	_,contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	#areas = [cv2.contourArea(c) for c in contours]
	#max_index = np.argmax(areas)
	#cnt=contours[max_index]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w > 50 and h > 50:
			#print("x = %4s   y = %4s   h = %4s   w = %4s"%(x, y, w, h))
			print((x, y, w, h))
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow("Show",im)
	cv2.waitKey(0)

def make_database():
	_, y, _, _ = cards_pos.main_field
	_, _, w, h = cards_pos.middle
	fig = plt.figure()
	y_delta = 31
	x_delta = 0
	y_diff = 8
	x_diff = 8
	x_size = 20
	y_size = 23
	for j, x in enumerate(cards_pos.main):
		for i in range(10):
			#cv2.imshow('image',img_rgb[y+y_diff:y+y_diff+y_size, x+x_diff:x+x_diff+x_size])
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			#cv2.imwrite("1.png", img_rgb[y+y_diff:y+y_diff+y_size, x+x_diff:x+x_diff+x_size])
			ax =fig.add_subplot(10, 8, i*8 + 1 + j)
			ax.imshow(img_rgb[y+y_diff+y_delta*i:y+y_diff+y_delta*i+y_size, x+x_diff+x_delta*i:x+x_diff+x_delta*i+x_size], 'gray')
			ax.autoscale(False)
			ax.axis('off')
		
	plt.show()
	
def read_database():
	global data
	imgs = glob.glob("data\*.png")
	bar_max, bar_i = len(imgs), 0	#progress bar
	for a in imgs:
		data[ tuple(int(x) for x in a[a.index('(')+1:a.index(')')].split(',')) ] = cv2.imread(a, 0)
		bar_i = bar_i + 1	#draw progressbar
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
	
	read_database()
	while AUTO_SOLVE == True:
		step = 0
		all_steps = 0
		solution_found = False
		map_dict = {}
		map_stack = {}
		map_checked = set()
		gc.collect()
		screen_map = Map_()
		screen_map.read_map()
		#screen_map = test_map2
		for w in move_weights:
			map_stack[w] = []
		map_stack[0].append(None)
		map_stack[50].append(screen_map)
		map_dict[screen_map] = []
		
		print("\nCurrent Map:\n")
		print(screen_map)
		print("\n")
		
		#click(50, 50)
		find_better_solution = True
		find_solution_path()
		if solution_found:
			while find_better_solution:
				all_moves, auto_moves = len(map_dict[win_map]), len([1 for m in map_dict[win_map] if m.weight == 50])
				print("\nSolution found: %s moves (%s manual, %s auto)"%(all_moves, all_moves - auto_moves, auto_moves))
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
							#print("Another solution found: same moves number")
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
		click(1115, 730)
		time.sleep(6)
	
	print("Program end.")
	