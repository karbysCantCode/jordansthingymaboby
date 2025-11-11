import random
import numpy

def generate_minefield(map_x, map_y, mine_count):
    minefield_array = numpy.zeros((map_x, map_y))
    
    total_cells = map_x * map_y
    if mine_count > total_cells:
      raise ValueError("Mine count cannot exceed total number of cells.")

    mines_to_place = mine_count
    while mines_to_place > 0:
      position_x = random.randint(0,map_x-1)
      position_y = random.randint(0,map_y-1)

      if minefield_array[position_x][position_y] == 0:
        minefield_array[position_x][position_y] = 1
        mines_to_place -= 1

    return minefield_array

# Example usage
MAP_X = 8
MAP_Y = 8
MINE_COUNT = 10

test_count = 10

verif_count = 0
for index in range(test_count):
  minefield = generate_minefield(MAP_X, MAP_Y, MINE_COUNT)

  # Print grid nicely
  for row in minefield:
      print('  '.join(str(int(cell)) for cell in row))

  mc = 0
  for row in minefield:
    for cell in row:
      mc += cell
  print(f'{mc} mines')

  if mc == MINE_COUNT:
    verif_count += 1

print(f'{verif_count}/{test_count} passed')