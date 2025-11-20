import random
import numpy
from math import prod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from enum import Enum
from tqdm import tqdm
from sympy import symbols, Symbol, Eq, solve, Add
from typing import Optional, Union
from collections import deque

#DEBUGGING
DEBUGGING = False
MUTE_INTERMEDIATE_PRINTS = True

#################################################################
# SETTINGS!!!!!!!!
#################################################################
MINE_DENSITY = 0.1                           #percentage as decimal, ie 10% = 0.1
MAP_X = 100                                   #the game width in cells
MAP_Y = 100                                   #the game height in cells
TEST_COUNT = 1000                         #the number of minesweeper games to attempt to play
SHOW_RESULT_OF_EACH_GAME = False             #will pop up a window showing the end of each match
#"comment lines" via adding a # before your "comment"
#the below line uses the mine density and map size to calculate a number of mines that is the density as specified
MINE_COUNT = int(MAP_X*MAP_Y*MINE_DENSITY)

#comment out the above "MINE_COUNT" and uncomment the below "MINE_COUNT" to select a specific number of mines
#MINE_COUNT = 10
#################################################################
# SETTINGS!!!!!!!!
#################################################################

#i'll make UI for this eventually :sob:









#below is code! i dont believe there are any valuable things to change down here for you, but let me know if youd like another variable!


DIRECTIONS = [
    (-1,  1), ( 0,  1), ( 1,  1),
    (-1,  0), ( 0,  0), ( 1,  0),
    (-1, -1), ( 0, -1), ( 1, -1)
  ]

def generate_minefield(map_x, map_y, mine_count):
    minefield_map = numpy.zeros((map_x, map_y), dtype=int)
    
    total_cells = map_x * map_y
    if mine_count > total_cells:
      raise ValueError("Mine count cannot exceed total number of cells.")

    start_position_x = random.randint(0,map_x-1)
    start_position_y = random.randint(0,map_y-1)

    invalid_mine_positions : list[tuple[int,int]] = []
    for dx,dy in DIRECTIONS:
      nx, ny = start_position_x + dx, start_position_y + dy
      invalid_mine_positions.append((nx,ny))


    mine_positions : list[tuple[int, int]] = []
    mines_to_place = mine_count
    while mines_to_place > 0:
      position_x = random.randint(0,map_x-1)
      position_y = random.randint(0,map_y-1)

      if (minefield_map[position_x][position_y] == 0) and (not ((position_x,position_y) in invalid_mine_positions)):
        minefield_map[position_x][position_y] = 1
        mines_to_place -= 1
        mine_positions.append((position_x,position_y))

    return minefield_map, start_position_x, start_position_y, mine_positions

import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def render_board(score_map, map_x, map_y, title="Minesweeper Board Visualization"):
    """
    Render a Minesweeper board from a 2D numpy array (score_map)
    using a consistent, unified color scheme.
    
    score_map: 2D numpy array containing:
       - numbers (0–8) for revealed counts
       - '*' for mines
       - 'N' for unrevealed tiles
    """

    # Convert score_map into numeric visualization array
    # -2 = mine, -1 = unrevealed ("N"), 0–8 = number
    numeric_board = numpy.full(score_map.shape, -1)
    for x in range(score_map.shape[0]):
        for y in range(score_map.shape[1]):
            cell = score_map[x, y]
            if isinstance(cell, (int, float)):
                numeric_board[x, y] = cell
            elif cell == "*":
                numeric_board[x, y] = -2  # mine

    # Unified color scheme:
    # Mines = dark red, Unrevealed = gray, Numbers = light→dark blue
    colors = [
        "#5a0000",  # -2 = mine
        "#b0b0b0",  # -1 = unrevealed
        "#e8f1fa",  # 0
        "#d3e4f3",  # 1
        "#bcd6ed",  # 2
        "#9ec3e2",  # 3
        "#7faed8",  # 4
        "#6098cd",  # 5
        "#3a7fc0",  # 6
        "#1f68b3",  # 7
        "#00489e"   # 8
    ]
    cmap = ListedColormap(colors)

    # Shift indices so -2→0, -1→1, 0→2, etc.
    plt.imshow(numeric_board + 2, cmap=cmap, origin="upper")

    # Grid overlay
    plt.gca().set_xticks(numpy.arange(-.5, map_y, 1), minor=True)
    plt.gca().set_yticks(numpy.arange(-.5, map_x, 1), minor=True)
    plt.grid(which="minor", color="black", linewidth=0.7)
    plt.tick_params(which="minor", size=0)
    plt.xticks([])
    plt.yticks([])

    # Text overlay (uniform black for readability)
    for (i, j), val in numpy.ndenumerate(score_map):
        if isinstance(val, (int, float)) and val != 0:
            plt.text(j, i, str(val), ha='center', va='center',
                     color='black', fontsize=11, weight='bold')
        elif val == "*":
            plt.text(j, i, "@", ha='center', va='center',
                     color='white', fontsize=11, weight='bold')

    plt.title(title, fontsize=14)
    plt.show()

def delta_text_map(start_x, start_y, map_x,map_y,score_map):
    for x in range(map_x):
      row = ''
      for y in range(map_y):
        if (x,y) in mine_positions:
          if score_map[x][y] == "*":
            row += f"  *"
          else:
            row += f"  X"
        elif (x,y) == (start_position_x, start_position_y):
          row += f"  !"
        else:
          row += f"  {score_map[x][y]}"

      print(row)









class mine_analysis_identity:
  def __init__(self,mines_around,flags_around_count,blank_cell_coordinates : list[tuple[int,int]]) -> None:

    #MINES_AROUND - FLAGS_AROUND
    self.mine_count = mines_around - flags_around_count

    #list of adjaent blanks
    self.potential_cell_coordinate_list : list[tuple[int,int]] = blank_cell_coordinates.copy()

    self.symbols : list[Symbol] = []
  def has_cell(self,coordinate : tuple[int,int]):
    if coordinate in self.potential_cell_coordinate_list:
      return True
    return False
  
  def construct_constraint(self,coordinate_to_symbol : dict[tuple[int,int],Symbol]):
    for cell in self.potential_cell_coordinate_list:
      self.symbols.append(coordinate_to_symbol[cell])
  











#solves from 0 spread
def solve_minefield(minefield_map, map_x, map_y, start_position_x, start_position_y,mine_positions):
  tags : set[completion_tags] = set()
  scored_map = numpy.full((map_x, map_y), 'N', dtype=object)
  already_spread_map = numpy.zeros((map_x, map_y), dtype=int)
  solved_cell_count = 0
  solved_cell_set : set[tuple[int,int]]= set()

  #will be init with the 0 spread discovered non 0's
  last_step_solved_map = numpy.zeros((map_x, map_y), dtype=int)

  changes_last_step = 1
  zero_spread_last_step_solved_coordinates : list[tuple[int, int]] = []
  zero_spread_last_step_solved_coordinates.append((start_position_x, start_position_y))

  #will be init with the 0 spread discovered non 0's
  last_step_solved_coordinates : list[tuple[int, int]] = []
  last_step_solved_coordinates.append((start_position_x, start_position_y))

  #ie discovered cells that still have undiscovered mines around them
  discovered_incomplete_coordinates : set[tuple[int, int]] = set()

  mines_around_cache : dict[tuple[int,int],int] = {}

  def get_mines_around(x,y):
    if (x,y) not in mines_around_cache:
      mines = 0
      for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map_x and 0 <= ny < map_y:
          if minefield_map[nx][ny] == 1:
            mines += 1
      mines_around_cache[(x,y)] = mines
      return mines
    return mines_around_cache[(x,y)]
  
  def get_flags_around(x,y):
    nonlocal scored_map
    flags = 0
    for dx, dy in DIRECTIONS:
      nx, ny = x + dx, y + dy
      if 0 <= nx < map_x and 0 <= ny < map_y:
        if scored_map[nx][ny] == "*":
          flags += 1
    return flags
  
  def get_empty_cells_around(x,y):
    nonlocal scored_map
    blanks = 0
    blank_coordinate_list : list[tuple[int,int]] = []
    for dx, dy in DIRECTIONS:
      nx, ny = x + dx, y + dy
      if 0 <= nx < map_x and 0 <= ny < map_y:
        if scored_map[nx][ny] == 'N':
          blanks += 1
          blank_coordinate_list.append((nx,ny))
    return blanks, blank_coordinate_list
  
  def zero_spread_from_position(position: tuple[int, int]):
    nonlocal solved_cell_set
    queue = deque([position])
    already_spread_set = set()  # track coordinates we've already processed

    while queue:
        x, y = queue.popleft()

        if (x, y) in already_spread_set:
            continue
        already_spread_set.add((x, y))

        current_location_mine_score = 0

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_x and 0 <= ny < map_y:
                if minefield_map[nx][ny] == 1:
                    current_location_mine_score += 1
                elif already_spread_map[nx][ny] == 0:
                    mine_count = get_mines_around(nx, ny)
                    if (nx, ny) not in already_spread_set and mine_count == 0:
                        queue.append((nx, ny))
                    else:
                        scored_map[nx][ny] = mine_count
                        discovered_incomplete_coordinates.add((nx, ny))  # use set for speed

        # Mark current cell as spread
        already_spread_map[x][y] = 1
        scored_map[x][y] = current_location_mine_score
  

  # 0 spread from init guess
  

  zero_spread_from_position((start_position_x,start_position_y))

  #solving!
  

  def basic_solve_loop():
    nonlocal changes_last_step
    nonlocal discovered_incomplete_coordinates
    nonlocal scored_map
    nonlocal solved_cell_set

    changes_last_step = len(discovered_incomplete_coordinates)

    while changes_last_step > 0:
      changes_last_step = 0
      last_square_unchanged = True

      for coordinate_tuple in discovered_incomplete_coordinates.copy():
        #debug
        x,y = coordinate_tuple[0],coordinate_tuple[1]
        if not last_square_unchanged and DEBUGGING:
          delta_text_map(start_position_x,start_position_y,map_x,map_y,scored_map)
          print()
          render_board(scored_map,MAP_X,MAP_Y)
        last_square_unchanged = True
        current_score = get_mines_around(x, y)
        blanks_around, blanks_coordinate_list = get_empty_cells_around(x, y)
        flags_around = get_flags_around(x, y)

        #solved all around by nearby's
        if blanks_around == 0:
          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)
          #debug
          last_square_unchanged = False
          continue
        
        #if there are only the number of blanks around or
        #if there are only the number of flag + blanks
        if current_score - flags_around == blanks_around:
          if not MUTE_INTERMEDIATE_PRINTS:
            print(f'{x},{y}, C={current_score}, F={flags_around}, B={blanks_around}')
          for x,y in blanks_coordinate_list:
            #flag blanks
            scored_map[x][y] = '*'

          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)
          #debug
          last_square_unchanged = False
          continue
        
        if flags_around == current_score and blanks_around > 0:
          for x,y in blanks_coordinate_list:
            mines_around_local = get_mines_around(x,y)
            scored_map[x][y] = mines_around_local
            if (x,y) not in discovered_incomplete_coordinates:
              if mines_around_local == 0:
                zero_spread_from_position((x,y))
              else:
                discovered_incomplete_coordinates.add((x,y))
              

          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)

          #debug
          last_square_unchanged = False
          continue
  def simultaneous_solver():
    identity_list : list[mine_analysis_identity] = []
    blanks_to_analyse : set[tuple[int,int]] = set()
    blank_symbol_to_coordinate : dict[Symbol, tuple[int,int]] = {}
    coordinate_to_blank_symbol : dict[tuple[int,int], Symbol] = {}

    for cell in discovered_incomplete_coordinates.copy():
      x,y = cell[0],cell[1]
      flags_around_cell = get_flags_around(x,y)
      mines_around_cell = get_mines_around(x,y)
      blank_around_list = get_empty_cells_around(x,y)[1]

      cell_identity = mine_analysis_identity(mines_around_cell,flags_around_cell,blank_around_list)
      identity_list.append(cell_identity)

      for blank in blank_around_list:
        if blank not in blanks_to_analyse:
          blanks_to_analyse.add(blank)
          symbol = symbols(str(len(blanks_to_analyse)), integer = True)
          blank_symbol_to_coordinate[symbol] = blank
          coordinate_to_blank_symbol[blank] = symbol

      cell_identity.construct_constraint(coordinate_to_blank_symbol)

    equations = []
    for identity in identity_list:
      expression = Add(*identity.symbols) # pyright: ignore[reportPossiblyUnboundVariable]
      equations.append(Eq(expression,identity.mine_count))

    solution = solve(equations, dict=True)

    if not solution:
      return False, None
    
    sol = solution[0]

    deduced = {}

    for var, val in sol.items():
      try:
        numeric = int(val)
        if numeric in (0, 1):
            deduced[blank_symbol_to_coordinate[var]] = numeric
      except:
          continue
      
    return len(deduced)>0, deduced
    


    

      

    



    



    

  def cell_probability_analysis():
    probability_frequencies : dict[float, int] = {}

    adjacent_unsolved_cells : dict[tuple[int,int],list[float]] = {}

    for cell in discovered_incomplete_coordinates:
      x = cell[0]
      y = cell[1]
      mine_count = get_mines_around(x,y) - get_flags_around(x,y)
      blank_count, blank_list = get_empty_cells_around(x,y)
      for blank in blank_list:
        if blank not in adjacent_unsolved_cells:
          adjacent_unsolved_cells[blank] = []
        adjacent_unsolved_cells[blank].append(1- (mine_count/blank_count))

    final_probability_by_cell : dict[tuple[int,int], float] = {}
    for cell in adjacent_unsolved_cells:
      final_probability = 1 - prod(adjacent_unsolved_cells[cell])
      final_probability_by_cell[cell] = final_probability
      if final_probability not in probability_frequencies:
        probability_frequencies[final_probability] = 0
      probability_frequencies[final_probability] += 1
    
    return final_probability_by_cell, probability_frequencies




  
  final_probability_by_cell : dict[tuple[int,int], float] = {}
  probability_frequencies : dict[float, int] = {}

  def main_solve_loop():
    nonlocal discovered_incomplete_coordinates
    nonlocal scored_map
    nonlocal final_probability_by_cell
    nonlocal probability_frequencies
    nonlocal mine_positions
    nonlocal solved_cell_set
    nonlocal map_x
    nonlocal map_y
    solved = False
    while not (solved):
      #print(solving_count)
      basic_solve_loop()
      #if there are unsolved cells
      if len(discovered_incomplete_coordinates) != 0:
        #print("a")
        simultaneous_resolved, deduced = simultaneous_solver()

        if simultaneous_resolved == False:
          tags.add(completion_tags.SIMULTANEOUS_FAILED_TO_RESOLVE)
          #print('b')
          #probability_frequencies is % to be a mine
          final_probability_by_cell,probability_frequencies = cell_probability_analysis()
          lowest_mine_probability = min(probability_frequencies)
          safest_coordinates = [k for k, v in final_probability_by_cell.items() if v == lowest_mine_probability]
          first_safest_coordinate = safest_coordinates[0]
          safest_x = first_safest_coordinate[0]
          safest_y = first_safest_coordinate[1]
          #IF 50/50
          if lowest_mine_probability == 0.5:
            #if final
            if len(solved_cell_set) + len(discovered_incomplete_coordinates) + len(final_probability_by_cell) + len(mine_positions) == map_x*map_y:
              # FINAL_STUCK_ONLY_AND_ONE_5050 = 5
              # FINAL_STUCK_MULTIPLE_5050 = 6
              # FINAL_STUCK_ONE_5050_AND_PROBABILITY_LESS_THAN_5050 = 9
              # FINAL_STUCK_MULTIPLE_5050_AND_PROBABILITY_LESS_THAN_5050 = 10
              if len(probability_frequencies) == 1:
                #only 5050's
                if probability_frequencies[0.5] == 2: #bc 5050 would involve 2 cells
                  #only onme
                  tags.add(completion_tags.FINAL_STUCK_ONLY_AND_ONE_5050)
                else:
                  #multiple
                  tags.add(completion_tags.FINAL_STUCK_MULTIPLE_5050)
              else:
                # ONE 5050 and lesser probabilities
                if probability_frequencies[0.5] == 2:#bc 5050 would involve 2 cells
                  #only onme 5050
                  tags.add(completion_tags.FINAL_STUCK_ONE_5050_AND_PROBABILITY_LESS_THAN_5050)
                else:
                  #multiple 5050
                  tags.add(completion_tags.FINAL_STUCK_MULTIPLE_5050_AND_PROBABILITY_LESS_THAN_5050)

            else:
              tags.add(completion_tags.MID_GAME_STUCK_5050)

            # for tag in tags:
            #   print(tag.name)
            
            # render_board(scored_map, map_x, map_y)
            solved = True
            break

          #ELSE #guess the safest (or one of the safest at random)
          if first_safest_coordinate in mine_positions:
            #IF IS BOMB
            tags.add(completion_tags.FAILED_BY_GUESSING_BOMB)
            solved = True
            break
          else:
            tags.add(completion_tags.SUCCESSFULLY_ELIMINATED_SOME_PROBABILITY_DURING_GAME)
            scored_map[safest_x][safest_y] = get_mines_around(safest_x,safest_y)
            discovered_incomplete_coordinates.add((safest_x,safest_y))



        elif simultaneous_resolved == True:
          tags.add(completion_tags.SIMULTANEOUS_SOLUTION_CONTRIBUTION)
          #if deduced == None: # type: ignore

          for coordinate, is_mine in deduced.items(): # type: ignore
            if DEBUGGING:
              print(f'{type(coordinate)}, {coordinate}, :: {type(is_mine)}, {is_mine}')
            #discovered_incomplete_coordinates.remove(coordinate)
            if is_mine:
              scored_map[coordinate[0]][coordinate[1]] = '*'
            else:
              scored_map[coordinate[0]][coordinate[1]] = get_mines_around(coordinate[0],coordinate[1])
              discovered_incomplete_coordinates.add((coordinate[0],coordinate[1]))
      else:
        solved = True

  
  main_solve_loop()





    
      


  return scored_map,discovered_incomplete_coordinates,final_probability_by_cell,probability_frequencies,tags, solved_cell_set

class completion_tags(Enum):
  INCOMPLETE = 0 #DONE
  INVALID_FLAG = 1 #DONE
  COMPLETE = 2 #DONE
  FAILED_BY_GUESSING_BOMB = 3 #DONE
  MID_GAME_STUCK_5050 = 4 #DINE
  FINAL_STUCK_ONLY_AND_ONE_5050 = 5
  FINAL_STUCK_MULTIPLE_5050 = 6
  FINAL_STUCK_ONE_5050_AND_PROBABILITY_LESS_THAN_5050 = 9
  FINAL_STUCK_MULTIPLE_5050_AND_PROBABILITY_LESS_THAN_5050 = 10
  SIMULTANEOUS_SOLUTION_CONTRIBUTION = 7 # done
  SIMULTANEOUS_FAILED_TO_RESOLVE = 8 #done
  SUCCESSFULLY_ELIMINATED_SOME_PROBABILITY_DURING_GAME = 11

class statistic:
  def __init__(self, count, base, message, name = '') -> None:
    self.count = count
    self.base = base
    self.message = message
    self.name = name

  def announce(self):
    print(f"{self.count}/{self.base} : {self.count/self.base*100}% {self.message}")

verif_count = 0
statistics : dict[str, statistic] = {}
for tag in completion_tags:
  statistics[tag.name] = statistic(0,0,'')

statistics[completion_tags.COMPLETE.name].base = TEST_COUNT
statistics[completion_tags.COMPLETE.name].message = "Games finished successfully."
statistics[completion_tags.INCOMPLETE.name].base = TEST_COUNT
statistics[completion_tags.INCOMPLETE.name].message = "Games finished incomplete."
statistics[completion_tags.INVALID_FLAG.name].base = TEST_COUNT
statistics[completion_tags.INVALID_FLAG.name].message = "Games finished with false flags."
statistics[completion_tags.FAILED_BY_GUESSING_BOMB.name].base = TEST_COUNT
statistics[completion_tags.FAILED_BY_GUESSING_BOMB.name].message = "Games failed eliminating probability. (mine selected)"
statistics[completion_tags.MID_GAME_STUCK_5050.name].base = TEST_COUNT
statistics[completion_tags.MID_GAME_STUCK_5050.name].message = "Games stuck on 50/50's but not as a final move"
statistics[completion_tags.FINAL_STUCK_ONLY_AND_ONE_5050.name].base = TEST_COUNT
statistics[completion_tags.FINAL_STUCK_ONLY_AND_ONE_5050.name].message = "Games stuck on ONE 50/50 AS a final move"
statistics[completion_tags.FINAL_STUCK_MULTIPLE_5050.name].base = TEST_COUNT
statistics[completion_tags.FINAL_STUCK_MULTIPLE_5050.name].message = "Games stuck on MULTIPLE and ONLY 50/50's as a final move"
statistics[completion_tags.FINAL_STUCK_ONE_5050_AND_PROBABILITY_LESS_THAN_5050.name].base = TEST_COUNT
statistics[completion_tags.FINAL_STUCK_ONE_5050_AND_PROBABILITY_LESS_THAN_5050.name].message = "Games stuck on ONE 50/50 and other probablities less than 50%% as final"
statistics[completion_tags.FINAL_STUCK_MULTIPLE_5050_AND_PROBABILITY_LESS_THAN_5050.name].base = TEST_COUNT
statistics[completion_tags.FINAL_STUCK_MULTIPLE_5050_AND_PROBABILITY_LESS_THAN_5050.name].message = "Games stuck on MULTIPLE 50/50's and other probablities less than 50%% as final"
statistics[completion_tags.SUCCESSFULLY_ELIMINATED_SOME_PROBABILITY_DURING_GAME.name].base = TEST_COUNT
statistics[completion_tags.SUCCESSFULLY_ELIMINATED_SOME_PROBABILITY_DURING_GAME.name].message = "Games that successfully eliminated probability during the round (may not have ended up completing, but did eliminate at least one successfully)"
statistics[completion_tags.SIMULTANEOUS_SOLUTION_CONTRIBUTION.name].base = TEST_COUNT
statistics[completion_tags.SIMULTANEOUS_SOLUTION_CONTRIBUTION.name].message = "Games where the simultaneous solver was used"
statistics[completion_tags.SIMULTANEOUS_FAILED_TO_RESOLVE.name].base = TEST_COUNT
statistics[completion_tags.SIMULTANEOUS_FAILED_TO_RESOLVE.name].message = "Games where the simultaneous solver was used but didnt help"

def verify_board(board, minefield, tags : set[completion_tags]):
  valid = True

  #cjecking all of board is explored
  x_pos = 0
  y_pos = 0
  for row in board:
    for cell in row:
      if cell == 'N':
        valid = False
        if completion_tags.INCOMPLETE not in tags:
          tags.add(completion_tags.INCOMPLETE)
      elif (cell == '*') != minefield[x_pos][y_pos]:
        valid = False
        if completion_tags.INVALID_FLAG not in tags:
          tags.add(completion_tags.INVALID_FLAG)
      y_pos += 1
    y_pos = 0
    x_pos += 1

        


  #checking invalid flags
  # for mine_coordinate in minefield:
  #   if not board[mine_coordinate[0]][mine_coordinate[1]] == '*':
  #     valid = False
  #     if completion_tags.INVALID_FLAG not in tags:
  #       tags.append(completion_tags.INVALID_FLAG)
  
  if valid:
    tags.add(completion_tags.COMPLETE)
  return valid, tags


class frequency_and_tag_wrapper:
  def __init__(self) -> None:
    self.frequency = 0
    self.tag_dict : dict[completion_tags,int] = {}

  def increment_frequency(self):
    self.frequency += 1
  
  def add_tags(self, tags : set[completion_tags]):
    for tag in tags:
      self.tag_dict.setdefault(tag,0)
      self.tag_dict[tag] += 1
  
  def announce_tags(self):
    for tag, frequency in self.tag_dict.items():
      print(f"    {tag.name} occured {frequency} times")


solved_cell_complete_delta_frequency : dict[int,frequency_and_tag_wrapper] = {} # delta : (frequency, tag : frequency)
for index in tqdm(range(TEST_COUNT),desc="games simulated", unit=" games"):
  minefield, start_position_x, start_position_y,mine_positions = generate_minefield(MAP_X, MAP_Y, MINE_COUNT)
  score_map,dico,final_probability_by_cell,probability_frequencies, tags,solved_cell_set = solve_minefield(minefield, MAP_X, MAP_Y, start_position_x, start_position_y,mine_positions)
  
  valid, tags = verify_board(score_map, minefield, tags)
  if completion_tags.COMPLETE in tags:
    key = MAP_X * MAP_Y - len(solved_cell_set)
    solved_cell_complete_delta_frequency.setdefault(key, frequency_and_tag_wrapper())
    solved_cell_complete_delta_frequency[key].increment_frequency()
    solved_cell_complete_delta_frequency[key].add_tags(tags)
    # if key != 10:
    #   print(key)
    #   render_board(score_map, MAP_X, MAP_Y)


  for tag in tags:
    statistics[tag.name].count += 1

  
  if SHOW_RESULT_OF_EACH_GAME:
    render_board(score_map,MAP_X,MAP_Y)

  if not MUTE_INTERMEDIATE_PRINTS:
    # Print grid nicely
    for row in minefield:
      print('  '.join(str(cell) for cell in row))
    print()
    for row in score_map:
      print('  '.join(str(cell) for cell in row))
    print()
    for x in range(MAP_X):
      row = ''
      for y in range(MAP_Y):
        if (x,y) in mine_positions:
          if score_map[x][y] == "*" and minefield[x][y] == 1:
            row += f"  *"
          else:
            row += f"  X"
        elif (x,y) == (start_position_x, start_position_y):
          row += f"  !"
        else:
          row += f"  {score_map[x][y]}"

      print(row)
        
     

  mc = 0
  for row in minefield:
    for cell in row:
      mc += cell
  if not MUTE_INTERMEDIATE_PRINTS:
    print(f'{mc} mines')

  if mc == MINE_COUNT:
    verif_count += 1

print(f'{verif_count}/{TEST_COUNT} valid mine placement')
for statistic_name, statistic_object in statistics.items():
  statistic_object.announce()

# for delta, frequency in solved_cell_complete_delta_frequency.items():
#   print(f"solved cell delta of {delta} occured {frequency.frequency} times")
#   frequency.announce_tags()