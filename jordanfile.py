import random
import numpy

# Example usage
MAP_X = 8
MAP_Y = 8
MINE_COUNT = 10

test_count = 1

def generate_minefield(map_x, map_y, mine_count):
    minefield_array = numpy.zeros((map_x, map_y))
    
    total_cells = map_x * map_y
    if mine_count > total_cells:
      raise ValueError("Mine count cannot exceed total number of cells.")

    start_position_x = random.randint(0,map_x-1)
    start_position_y = random.randint(0,map_y-1)

    mines_to_place = mine_count
    while mines_to_place > 0:
      position_x = random.randint(0,map_x-1)
      position_y = random.randint(0,map_y-1)

      if (minefield_array[position_x][position_y] == 0) and (position_x != start_position_x) and (position_y != start_position_y):
        minefield_array[position_x][position_y] = 1
        mines_to_place -= 1

    return minefield_array, start_position_x, start_position_y

#solves from 0 spread
def solve_minefield(minefield_array, map_x, map_y, start_position_x, start_position_y):
  is_5050 = False
  mutated_minefield = minefield_array
  scored_map = numpy.zeros((map_x, map_y))

  already_spread_map = numpy.zeros((map_x, map_y))

  #will be init with the 0 spread discovered non 0's
  last_step_solved_map = numpy.zeros((map_x, map_y))

  changes_last_step = 1
  zero_spread_last_step_solved_coordinates : list[tuple[int, int]] = []
  zero_spread_last_step_solved_coordinates.append((start_position_x, start_position_y))

  #will be init with the 0 spread discovered non 0's
  last_step_solved_coordinates : list[tuple[int, int]] = []
  last_step_solved_coordinates.append((start_position_x, start_position_y))

  DIRECTIONS = [
    (-1,  1), ( 0,  1), ( 1,  1),
    (-1,  0),           ( 1,  0),
    (-1, -1), ( 0, -1), ( 1, -1)
  ]
  # 0 spread from init guess
  while changes_last_step > 0:
    changes_last_step = 0
    next_step_coordinates : list[tuple[int, int]] = []



    #where x = [0] and y = [1]
    for coordinate_tuple in zero_spread_last_step_solved_coordinates:
      current_location_mine_score = 0

      up_available = coordinate_tuple[1] < (map_y - 1)
      down_available = coordinate_tuple[1] > 0

      x_plus_one = coordinate_tuple[0]+1
      x_minus_one = coordinate_tuple[0]-1
      y_plus_one = coordinate_tuple[1]+1
      y_minus_one = coordinate_tuple[1]-1

      #left checking
      if coordinate_tuple[0] > 0:
        # top left
        if up_available:
          if minefield_array[x_minus_one][y_plus_one] == 1:
            current_location_mine_score += 1
          elif 0 == already_spread_map[x_minus_one][y_plus_one]:
            next_step_coordinates.append((x_minus_one,y_plus_one))
            changes_last_step += 1
        
        # down left
        if down_available:
          if minefield_array[x_minus_one][y_minus_one] == 1:
            current_location_mine_score += 1
          elif 0 == already_spread_map[x_minus_one][y_minus_one]:
            next_step_coordinates.append((x_minus_one,y_minus_one))
            changes_last_step += 1

        #immediate left
        if minefield_array[x_minus_one][coordinate_tuple[1]] == 1:
          current_location_mine_score += 1
        elif 0 == already_spread_map[x_minus_one][coordinate_tuple[1]]:
          next_step_coordinates.append((x_minus_one,coordinate_tuple[1]))
          changes_last_step += 1



      #right checking
      if coordinate_tuple[0] < (map_x - 1):
        # top right
        if up_available:
          if minefield_array[x_plus_one][y_plus_one] == 1:
            current_location_mine_score += 1
          elif 0 == already_spread_map[x_plus_one][y_plus_one]:
            next_step_coordinates.append((x_plus_one,y_plus_one))
            changes_last_step += 1
        
        # down right
        if down_available:
          if minefield_array[x_plus_one][y_minus_one] == 1:
            current_location_mine_score += 1
          elif 0 == already_spread_map[x_plus_one][y_minus_one]:
            next_step_coordinates.append((x_plus_one,y_minus_one))
            changes_last_step += 1

        if minefield_array[x_plus_one][coordinate_tuple[1]] == 1:
          current_location_mine_score += 1
        elif 0 == already_spread_map[x_plus_one][coordinate_tuple[1]]:
          next_step_coordinates.append((x_plus_one,coordinate_tuple[1]))
          changes_last_step += 1

      #immediate up
      if up_available:
        if minefield_array[coordinate_tuple[0]][y_plus_one] == 1:
          current_location_mine_score += 1
        elif 0 == already_spread_map[coordinate_tuple[0]][y_plus_one]:
          next_step_coordinates.append((coordinate_tuple[0],y_plus_one))
          changes_last_step += 1

      #immediate down
      if down_available:
        if minefield_array[coordinate_tuple[0]][y_minus_one] == 1:
          current_location_mine_score += 1
        elif 0 == already_spread_map[coordinate_tuple[0]][y_minus_one]:
          next_step_coordinates.append((coordinate_tuple[0],y_minus_one))
          changes_last_step += 1


      #to prepare non-zero cells to be solved by the normal solving algorithm
      if current_location_mine_score > 0:
        last_step_solved_coordinates.append((coordinate_tuple[0], coordinate_tuple[1]))

      already_spread_map[coordinate_tuple[0]][coordinate_tuple[1]] = 1
      scored_map[coordinate_tuple[0]][coordinate_tuple[1]] = current_location_mine_score
      print(f"{coordinate_tuple[0]},{coordinate_tuple[1]} = {current_location_mine_score}")



    zero_spread_last_step_solved_coordinates.clear()
    for next_cell in next_step_coordinates:
      print(f"to check {next_cell[0]},{next_cell[1]}")
    print(f"{changes_last_step} changed")
    zero_spread_last_step_solved_coordinates = next_step_coordinates
    next_step_coordinates.clear()


  return mutated_minefield, is_5050, scored_map




verif_count = 0
for index in range(test_count):
  minefield, start_position_x, start_position_y = generate_minefield(MAP_X, MAP_Y, MINE_COUNT)
  solved_minefield, is_5050, score_map = solve_minefield(minefield, MAP_X, MAP_Y, start_position_x, start_position_y)
  # Print grid nicely
  for row in minefield:
      print('  '.join(str(int(cell)) for cell in row))
  print()
  for row in score_map:
      print('  '.join(str(int(cell)) for cell in row))

  mc = 0
  for row in minefield:
    for cell in row:
      mc += cell
  print(f'{mc} mines')

  if mc == MINE_COUNT:
    verif_count += 1

print(f'{verif_count}/{test_count} passed')