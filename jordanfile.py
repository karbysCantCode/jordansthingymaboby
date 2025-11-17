import random
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from enum import Enum
from tqdm import tqdm
from sympy import symbols, Symbol, Eq, solve, Add
from typing import Optional, Union

#DEBUGGING
DEBUGGING = False
MUTE_INTERMEDIATE_PRINTS = True

#################################################################
# SETTINGS!!!!!!!!
#################################################################
MINE_DENSITY = 0.1                           #percentage as decimal, ie 10% = 0.1
MAP_X = 10                                   #the game width in cells
MAP_Y = 10                                   #the game height in cells
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
    minefield_array = numpy.zeros((map_x, map_y), dtype=int)
    
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

      if (minefield_array[position_x][position_y] == 0) and (not ((position_x,position_y) in invalid_mine_positions)):
        minefield_array[position_x][position_y] = 1
        mines_to_place -= 1
        mine_positions.append((position_x,position_y))

    return minefield_array, start_position_x, start_position_y, mine_positions

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
def solve_minefield(minefield_array, map_x, map_y, start_position_x, start_position_y):
  is_5050 = False
  mutated_minefield = minefield_array
  scored_map = numpy.full((map_x, map_y), 'N', dtype=object)
  bombed_map = numpy.full((map_x, map_y), 'N', dtype=object)
  already_spread_map = numpy.zeros((map_x, map_y), dtype=int)

  #will be init with the 0 spread discovered non 0's
  last_step_solved_map = numpy.zeros((map_x, map_y), dtype=int)

  changes_last_step = 1
  zero_spread_last_step_solved_coordinates : list[tuple[int, int]] = []
  zero_spread_last_step_solved_coordinates.append((start_position_x, start_position_y))

  #will be init with the 0 spread discovered non 0's
  last_step_solved_coordinates : list[tuple[int, int]] = []
  last_step_solved_coordinates.append((start_position_x, start_position_y))

  #ie discovered cells that still have undiscovered mines around them
  discovered_incomplete_coordinates : list[tuple[int, int]] = []

  def get_mines_around(x,y):
    nonlocal scored_map
    mines = 0
    for dx, dy in DIRECTIONS:
      nx, ny = x + dx, y + dy
      if 0 <= nx < map_x and 0 <= ny < map_y:
        if minefield_array[nx][ny] == 1:
          mines += 1
          bombed_map[nx][ny] = "*"
    return mines
  
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
  
  def zero_spread_from_position(position : tuple[int,int]):
    changes_last_step = 1
    zero_spread_last_step_solved_coordinates : list[tuple[int, int]] = []
    zero_spread_last_step_solved_coordinates.append(position)

    while changes_last_step > 0:
      changes_last_step = 0
      next_step_coordinates : list[tuple[int, int]] = []



      #where x = [0] and y = [1]
      for coordinate_tuple in zero_spread_last_step_solved_coordinates:
        current_location_mine_score = 0
        for dx, dy in DIRECTIONS:
          nx, ny = coordinate_tuple[0] + dx, coordinate_tuple[1] + dy
          if 0 <= nx < map_x and 0 <= ny < map_y:
            if minefield_array[nx][ny] == 1:
                current_location_mine_score += 1
                bombed_map[nx][ny] = "*"
            elif already_spread_map[nx][ny] == 0:
              mine_count = get_mines_around(nx, ny)
              if mine_count == 0:
                if (nx,ny) not in next_step_coordinates:
                  next_step_coordinates.append((nx, ny))
                  changes_last_step += 1
              else:
                scored_map[nx][ny] = mine_count
                bombed_map[nx][ny] = mine_count
                if (nx,ny) not in discovered_incomplete_coordinates:
                  discovered_incomplete_coordinates.append((nx,ny))

        already_spread_map[coordinate_tuple[0]][coordinate_tuple[1]] = 1
        scored_map[coordinate_tuple[0]][coordinate_tuple[1]] = current_location_mine_score
        bombed_map[coordinate_tuple[0]][coordinate_tuple[1]] = current_location_mine_score



      zero_spread_last_step_solved_coordinates.clear()
      zero_spread_last_step_solved_coordinates = next_step_coordinates.copy()
      next_step_coordinates.clear()
  

  # 0 spread from init guess
  

  zero_spread_from_position((start_position_x,start_position_y))

  #solving!
  

  def basic_solve_loop():
    nonlocal changes_last_step
    nonlocal discovered_incomplete_coordinates
    nonlocal scored_map

    changes_last_step = len(discovered_incomplete_coordinates)

    while changes_last_step > 0:
      changes_last_step = 0
      last_square_unchanged = True

      for coordinate_tuple in discovered_incomplete_coordinates.copy():
        #debug
        if not last_square_unchanged and DEBUGGING:
          delta_text_map(start_position_x,start_position_y,map_x,map_y,scored_map)
          print()
          render_board(scored_map,MAP_X,MAP_Y)
        last_square_unchanged = True
        current_score = get_mines_around(coordinate_tuple[0], coordinate_tuple[1])
        blanks_around, blanks_coordinate_list = get_empty_cells_around(coordinate_tuple[0], coordinate_tuple[1])
        flags_around = get_flags_around(coordinate_tuple[0], coordinate_tuple[1])

        #solved all around by nearby's
        if blanks_around == 0:
          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          #debug
          last_square_unchanged = False
          continue
        
        #if there are only the number of blanks around or
        #if there are only the number of flag + blanks
        if ((current_score == blanks_around) and (0 == flags_around)) or (current_score - flags_around == blanks_around):
          if not MUTE_INTERMEDIATE_PRINTS:
            print(f'{coordinate_tuple[0]},{coordinate_tuple[1]}, C={current_score}, F={flags_around}, B={blanks_around}')
          for blank_coordinate_tuple in blanks_coordinate_list:
            #flag blanks
            scored_map[blank_coordinate_tuple[0]][blank_coordinate_tuple[1]] = '*'

          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          #debug
          last_square_unchanged = False
          continue
        
        if flags_around == current_score and blanks_around > 0:
          for blank_coordinate_tuple in blanks_coordinate_list:
            mines_around_local = get_mines_around(blank_coordinate_tuple[0],blank_coordinate_tuple[1])
            scored_map[blank_coordinate_tuple[0]][blank_coordinate_tuple[1]] = mines_around_local
            if blank_coordinate_tuple not in discovered_incomplete_coordinates:
              if mines_around_local == 0:
                zero_spread_from_position(blank_coordinate_tuple)
              else:
                discovered_incomplete_coordinates.append((blank_coordinate_tuple[0],blank_coordinate_tuple[1]))
              

          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)

          #debug
          last_square_unchanged = False
          continue
  def simultaneous_solver():
    identity_list : list[mine_analysis_identity] = []
    blanks_to_analyse : list[tuple[int,int]] = []
    blank_symbol_to_coordinate : dict[Symbol, tuple[int,int]] = {}
    coordinate_to_blank_symbol : dict[tuple[int,int], Symbol] = {}

    for cell in discovered_incomplete_coordinates.copy():
      flags_around_cell = get_flags_around(cell[0],cell[1])
      mines_around_cell = get_mines_around(cell[0],cell[1])
      blank_around_count, blank_around_list = get_empty_cells_around(cell[0],cell[1])

      cell_identity = mine_analysis_identity(mines_around_cell,flags_around_cell,blank_around_list)
      identity_list.append(cell_identity)

      for blank in blank_around_list:
        if blank not in blanks_to_analyse:
          blanks_to_analyse.append(blank)
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
    pass
  
  not_sim_res = False
  simultaneous_solution_contribution = False
  def main_solve_loop():
    nonlocal discovered_incomplete_coordinates
    nonlocal scored_map
    nonlocal simultaneous_solution_contribution
    nonlocal not_sim_res
    solved = False
    while not (solved):
      #print(solving_count)
      basic_solve_loop()
      #if there are unsolved cells
      if len(discovered_incomplete_coordinates) != 0:
        #print("a")
        simultaneous_resolved, deduced = simultaneous_solver()

        if simultaneous_resolved == False:
          not_sim_res = True
          #print('b')
          cell_coordinate_by_probability = cell_probability_analysis()
          #check if 50/50's
          solved = True
          break
        elif simultaneous_resolved == True:
          simultaneous_solution_contribution = True
          #if deduced == None: # type: ignore

          for coordinate, is_mine in deduced.items(): # type: ignore
            if DEBUGGING:
              print(f'{type(coordinate)}, {coordinate}, :: {type(is_mine)}, {is_mine}')
            #discovered_incomplete_coordinates.remove(coordinate)
            if is_mine:
              scored_map[coordinate[0]][coordinate[1]] = '*'
            else:
              scored_map[coordinate[0]][coordinate[1]] = get_mines_around(coordinate[0],coordinate[1])
              discovered_incomplete_coordinates.append((coordinate[0],coordinate[1]))
      else:
        solved = True

  
  main_solve_loop()





    
      


  return mutated_minefield, is_5050, scored_map,discovered_incomplete_coordinates, simultaneous_solution_contribution, not_sim_res

class completion_tags(Enum):
  INCOMPLETE = 0
  INVALID_FLAG = 1
  COMPLETE = 2



  

def verify_board(board, minefield):
  valid = True
  tags : list[completion_tags] = []

  #cjecking all of board is explored
  x_pos = 0
  y_pos = 0
  for row in board:
    for cell in row:
      if cell == 'N':
        valid = False
        if completion_tags.INCOMPLETE not in tags:
          tags.append(completion_tags.INCOMPLETE)
      elif (cell == '*') != minefield[x_pos][y_pos]:
        valid = False
        if completion_tags.INVALID_FLAG not in tags:
          tags.append(completion_tags.INVALID_FLAG)
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
    tags.append(completion_tags.COMPLETE)
  return valid, tags



verif_count = 0
completion_counts = {
  "COMPLETE" : 0,
  "INVALID_FLAG" : 0,
  "INCOMPLETE" : 0,
  "COMPLETE_WITH_ZERO_UNDISCOVERED" : 0,
  "SIMULTANEOUS_SOLVER_CONTRIBUTION" : 0,
  "SIMULTANEOUS_SOLVER_FAILED_AT_LEAST_ONCE" : 0
}

for index in tqdm(range(TEST_COUNT),desc="games simulated", unit=" games"):
  minefield, start_position_x, start_position_y,mine_positions = generate_minefield(MAP_X, MAP_Y, MINE_COUNT)
  solved_minefield, is_5050, score_map,dico,sim_sol_contr, not_sim_res = solve_minefield(minefield, MAP_X, MAP_Y, start_position_x, start_position_y)
  
  valid, tags = verify_board(score_map, minefield)
  if completion_tags.COMPLETE in tags:
    completion_counts["COMPLETE"] += 1
    if len(dico) == 0:
      completion_counts["COMPLETE_WITH_ZERO_UNDISCOVERED"] += 1
  if completion_tags.INVALID_FLAG in tags:
    completion_counts["INVALID_FLAG"] += 1
  if completion_tags.INCOMPLETE in tags:
    completion_counts["INCOMPLETE"] += 1
  if sim_sol_contr:
    completion_counts["SIMULTANEOUS_SOLVER_CONTRIBUTION"] += 1
  if not_sim_res:
    completion_counts["SIMULTANEOUS_SOLVER_FAILED_AT_LEAST_ONCE"] += 1
  
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
print(f"{completion_counts['COMPLETE']}/{TEST_COUNT}, {completion_counts['COMPLETE']/TEST_COUNT*100}% completed successfully")
print(f"{completion_counts['COMPLETE_WITH_ZERO_UNDISCOVERED']}/{TEST_COUNT}, {completion_counts['COMPLETE_WITH_ZERO_UNDISCOVERED']/completion_counts['COMPLETE']*100}% of complete games completed without undiscovered successfully")
print(f"{completion_counts['SIMULTANEOUS_SOLVER_CONTRIBUTION']}/{TEST_COUNT}, {completion_counts['SIMULTANEOUS_SOLVER_CONTRIBUTION']/TEST_COUNT*100}% used simultaneous")

print(f"{completion_counts['INCOMPLETE']}/{TEST_COUNT}, {completion_counts['INCOMPLETE']/TEST_COUNT*100}% finished incomplete")
print(f"{completion_counts['INVALID_FLAG']}/{TEST_COUNT}, {completion_counts['INVALID_FLAG']/TEST_COUNT*100}% finished with invalid flags")
print(f"{completion_counts['SIMULTANEOUS_SOLVER_FAILED_AT_LEAST_ONCE']}/{TEST_COUNT}, {completion_counts['SIMULTANEOUS_SOLVER_FAILED_AT_LEAST_ONCE']/TEST_COUNT*100}% SIMULTANEOUS FAILED")