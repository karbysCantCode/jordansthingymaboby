from __future__ import annotations
import random
import threading
import time
from tkinter import messagebox
import numpy
from math import prod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from enum import Enum
from tqdm import tqdm
from sympy import symbols, Symbol, Eq, solve, Add
from typing import Optional, Union, Tuple, List, Callable
from collections import deque
import csv
import tkinter as tk
import pygame
from tkinter import ttk
from tkinter import filedialog
from dataclasses import dataclass
from copy import deepcopy
import math
import struct
import zlib
import os


#################################################################
# SETTINGS!!!!!!!!
#################################################################
SAVEFILEMAGIC = b"MSWPSSF"

#HIGHKEY THESE DONT DO ANYTHING ANYMORE BC OF THE UI


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
assert(False)
#IMPLEMENT the probability analysis when NO DICO and unsolved cells, just by collated probablility or somethingg

MAPCOLORS = [
    (0xe8, 0xf1, 0xfa),      # 0
    (0xd3, 0xe4, 0xf3),      # 1
    (0xbc, 0xd6, 0xed),      # 2
    (0x9e, 0xc3, 0xe2),      # 3
    (0x7f, 0xae, 0xd8),      # 4
    (0x60, 0x98, 0xcd),      # 5
    (0x3a, 0x7f, 0xc0),      # 6
    (0x1f, 0x68, 0xb3),      # 7
    (0x00, 0x48, 0x9e),      # 8
    (0x5a, 0x00, 0x00),      # -2 = mine
    (0xb0, 0xb0, 0xb0),      # -1 = unrevealed
    (0xff, 0xff, 0x00)
]

REVERSEMAPCOLORS = {
  (0xe8, 0xf1, 0xfa) : 0,
  (0xd3, 0xe4, 0xf3) : 1,
  (0xbc, 0xd6, 0xed) : 2,
  (0x9e, 0xc3, 0xe2) : 3,
  (0x7f, 0xae, 0xd8) : 4,
  (0x60, 0x98, 0xcd) : 5,
  (0x3a, 0x7f, 0xc0) : 6,
  (0x1f, 0x68, 0xb3) : 7,
  (0x00, 0x48, 0x9e) : 8,
  (0x5a, 0x00, 0x00) : 9,
  (0xb0, 0xb0, 0xb0) : 10,
  (0xff, 0xff, 0x00) : 11
}






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
  





Color = Tuple[int, int, int]

# @dataclass
# class Cell:
#   color: Color
#   text: Optional[str] = None

#   def copy(self):
#         return Cell(self.color, self.text)

# @dataclass
# class BoardState:
#     cells: List[List[Cell]]  # cells[y][x]
#     label: str

#     def copy(self):
#         return BoardState(
#             cells=[[cell.copy() for cell in row] for row in self.cells],
#             label=self.label
#         )
    
@dataclass
class BoardChange:
  coordinate: Tuple[int,int]
  color: Color
  cell_text: str
  change_text: str

@dataclass
class RenderCell:
    color: Color
    text: str


class MinesweeperRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        cell_size: int = 32,
        header_height: int = 40,
        font_size: int = 10
    ):
        pygame.init()

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.header_height = header_height

        self.screen = pygame.display.set_mode((
            width * cell_size,
            height * cell_size + header_height
        ))

        pygame.display.set_caption("Minesweeper Solver Viewer")

        self.font = pygame.font.SysFont("consolas", font_size)
        self.header_font = pygame.font.SysFont("consolas", font_size + 2, bold=True)
        self.clock = pygame.time.Clock()

        self.board: list[list[RenderCell]] = []
        self.header_text: str = ""

        self.last_change: Optional[BoardChange] = None

    def reset_board(self, round:round):
        self.board = [
            [
                RenderCell(MAPCOLORS[10], "")
                for cell in range(round.simulation.height)
            ]
            for column in range(round.simulation.width)
        ]
        self.header_text = "Starting position"
        self.last_change = None
        self.board[round.start_position[0]][round.start_position[1]] = RenderCell(MAPCOLORS[0], "0")

    def change(self, change: BoardChange, is_special = False):
      x, y = change.coordinate
      if not is_special:
        self.board[x][y].color = change.color
      else:
        self.board[x][y].color = (255,255,0)
      self.board[x][y].text = change.cell_text
      self.header_text = change.change_text

    def apply_changes(self, changes: List[BoardChange]):
      if self.last_change:
        self.change(self.last_change)
      change_c = 0
      while change_c < len(changes) - 1:
        self.change(changes[change_c])
        change_c += 1
      self.last_change = changes[-1]
      self.change(self.last_change, True)
    
    def draw(self):
      self.screen.fill((30, 30, 30))

      header_surf = self.header_font.render(
        self.header_text, True, (220, 220, 220)
      )
      self.screen.blit(header_surf, (8, 8))

      for x in range(self.width):
        for y in range(self.height):
          cell = self.board[x][y]
          px = x * self.cell_size
          py = y * self.cell_size + self.header_height

          rect = pygame.Rect(px, py, self.cell_size, self.cell_size)
          pygame.draw.rect(self.screen, cell.color, rect)
          pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

          if cell.text:
            text_surf = self.font.render(cell.text, True, (0, 0, 0))
            self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

        pygame.display.flip()

    def run_round(self, rnd:round):
      index = 0
      self.reset_board(rnd)

      self.draw()

      running = True
      while running:
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
              running = False

            elif event.type == pygame.KEYDOWN:
              if event.key == pygame.K_ESCAPE:
                running = False

              elif event.key == pygame.K_SPACE:
                index = min(index + 1, len(rnd.changes) - 1)
                self.apply_changes([rnd.changes[index]])
                self.draw()

              elif event.key == pygame.K_BACKSPACE:
                index = max(index - 1, 0)
                self.reset_board(rnd)
                self.apply_changes(rnd.changes[0:index+1])
                self.draw()

      pygame.quit()






#solves from 0 spread
def solve_minefield(minefield_map, map_x : int, map_y : int, start_position_x : int, start_position_y : int,mine_positions : list[tuple[int, int]]):
  changes: List[BoardChange] = []
  
  tags : set[completion_tags] = set()
  scored_map = numpy.full((map_x, map_y), 'N', dtype=object)
  already_spread_map = numpy.zeros((map_x, map_y), dtype=int)
  solved_cell_set : set[tuple[int,int]]= set()

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
                        change = BoardChange((nx,ny),
                                             MAPCOLORS[mine_count],
                                             str(mine_count),
                                             "Uncovered by zero spreading")
                        changes.append(change)
                        already_spread_set.add((x, y))


        # Mark current cell as spread
        already_spread_map[x][y] = 1
        already_spread_set.add((x, y))
        scored_map[x][y] = current_location_mine_score
        change = BoardChange((x,y),
              MAPCOLORS[current_location_mine_score],
              str(current_location_mine_score),
              "Uncovered by zero spreading")
        changes.append(change)
  

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

      for coordinate_tuple in discovered_incomplete_coordinates.copy():
        x,y = coordinate_tuple[0],coordinate_tuple[1]
        current_score = get_mines_around(x, y)
        blanks_around, blanks_coordinate_list = get_empty_cells_around(x, y)
        flags_around = get_flags_around(x, y)

        #solved all around by nearby's
        if blanks_around == 0:
          changes_last_step += 1
          change = BoardChange(coordinate_tuple,
              MAPCOLORS[flags_around],
              str(flags_around),
              "Solved All Around (debug: 1)")
          changes.append(change)
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)
          continue
        
        #if there are only the number of blanks around or
        #if there are only the number of flag + blanks
        if current_score - flags_around == blanks_around:
          for x2,y2 in blanks_coordinate_list:
            #flag blanks
            change = BoardChange((x2,y2),
              MAPCOLORS[9],
              "*",
              "Solved By Neighboring Flag Count")
            changes.append(change)
            scored_map[x2][y2] = '*'
            solved_cell_set.add((x2,y2))

          m_around = get_mines_around(x,y)
          change = BoardChange(coordinate_tuple,
              MAPCOLORS[m_around],
              str(m_around),
              "Solved All Around 2")
          changes.append(change)
          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)
          continue
        
        if flags_around == current_score and blanks_around > 0:
          for x2,y2 in blanks_coordinate_list:
            mines_around_local = get_mines_around(x2,y2)
            scored_map[x2][y2] = mines_around_local
            change = BoardChange((x2,y2),
                MAPCOLORS[mines_around_local],
                str(mines_around_local),
                "Sufficient flag count by neighbor")
            changes.append(change)
            if (x2,y2) not in discovered_incomplete_coordinates:
              if mines_around_local == 0:
                zero_spread_from_position((x2,y2))
              else:
                discovered_incomplete_coordinates.add((x2,y2))
          
          m_around = get_mines_around(x,y)
          change = BoardChange(coordinate_tuple,
                MAPCOLORS[m_around],
                str(m_around),
                "Sufficient flag count")
          changes.append(change)
          changes_last_step += 1
          discovered_incomplete_coordinates.remove(coordinate_tuple)
          solved_cell_set.add(coordinate_tuple)
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
    @dataclass
    class Cell_Probability_Identity:
      contributors: int = 0
      summed_probability: float = 0

      def get_probability(self):
        return self.summed_probability/self.contributors
      
      def add_probability(self,probability:float):
        self.contributors += 1
        self.summed_probability += probability

    probability_frequencies : dict[float, int] = {}

    adjacent_unsolved_cells : dict[tuple[int,int],Cell_Probability_Identity] = {}

    for cell in discovered_incomplete_coordinates:
      x = cell[0]
      y = cell[1]
      mine_count = get_mines_around(x,y) - get_flags_around(x,y)
      blank_count, blank_list = get_empty_cells_around(x,y)
      for blank in blank_list:
        if blank not in adjacent_unsolved_cells:
          adjacent_unsolved_cells[blank] = Cell_Probability_Identity()
        adjacent_unsolved_cells[blank].add_probability(1- (mine_count/blank_count))

    final_probability_by_cell : dict[tuple[int,int], float] = {}
    for cell, identity in adjacent_unsolved_cells.items():
      final_probability = 1 - identity.get_probability()
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
          for cell, probability in final_probability_by_cell.items():
            change = BoardChange(cell,
              (255,255,0),
              f"{probability*100:.1f}%",
              "Probability Analysis Displaying...")
            changes.append(change)
          for cell, probability in final_probability_by_cell.items():
            change = BoardChange(cell,
              MAPCOLORS[10],
              "",
              "Probability Analysis Cleaning...")
            changes.append(change)
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
            change = BoardChange(first_safest_coordinate,
              MAPCOLORS[10],
              "MINE",
              "Opened mine by probability guessing")
            changes.append(change)
            break
          else:
            tags.add(completion_tags.SUCCESSFULLY_ELIMINATED_SOME_PROBABILITY_DURING_GAME)
            mine_count = get_mines_around(safest_x,safest_y)
            scored_map[safest_x][safest_y] = mine_count
            discovered_incomplete_coordinates.add((safest_x,safest_y))
            change = BoardChange(first_safest_coordinate,
              MAPCOLORS[mine_count],
              str(mine_count),
              "Guessed correctly by probability")
            changes.append(change)



        elif simultaneous_resolved == True:
          tags.add(completion_tags.SIMULTANEOUS_SOLUTION_CONTRIBUTION)
          #if deduced == None: # type: ignore

          for coordinate, is_mine in deduced.items(): # type: ignore
            if is_mine:
              scored_map[coordinate[0]][coordinate[1]] = '*'
              change = BoardChange(coordinate,
                MAPCOLORS[9],
                "*",
                "Mine solved by simultaneous")
              changes.append(change)
            else:
              mines_around = get_mines_around(coordinate[0],coordinate[1])
              change = BoardChange(coordinate,
                MAPCOLORS[mines_around],
                str(mines_around),
                "Cell solved by simultaneous")
              changes.append(change)
              scored_map[coordinate[0]][coordinate[1]] = mines_around
              discovered_incomplete_coordinates.add((coordinate[0],coordinate[1]))
      else:
        #get all tuples that are blank
        undiscovered_cell_set: set[Tuple[int,int]] = set()
        x = 0
        for row in scored_map:
          y = 0
          for cell in row:
            if (cell == "N"):
              undiscovered_cell_set.add((x,y))
            y += 1
          x += 1
        solved = True

  
  main_solve_loop()





    
      


  return scored_map,discovered_incomplete_coordinates,final_probability_by_cell,probability_frequencies,tags, solved_cell_set, changes

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


def example_usage():
  verif_count = 0
  solved_cell_complete_delta_frequency : dict[int,frequency_and_tag_wrapper] = {} # delta : (frequency, tag : frequency)
  for index in tqdm(range(TEST_COUNT),desc="games simulated", unit=" games"):
    minefield, start_position_x, start_position_y,mine_positions = generate_minefield(MAP_X, MAP_Y, MINE_COUNT)
    score_map,dico,final_probability_by_cell,probability_frequencies, tags,solved_cell_set, states = solve_minefield(minefield, MAP_X, MAP_Y, start_position_x, start_position_y,mine_positions)
    
    valid, tags = verify_board(score_map, minefield, tags)
    if completion_tags.COMPLETE in tags:
      key = MAP_X * MAP_Y - len(solved_cell_set)
      solved_cell_complete_delta_frequency.setdefault(key, frequency_and_tag_wrapper())
      solved_cell_complete_delta_frequency[key].increment_frequency()
      solved_cell_complete_delta_frequency[key].add_tags(tags)


    for tag in tags:
      statistics[tag.name].count += 1

    
    if SHOW_RESULT_OF_EACH_GAME:
      render_board(score_map,MAP_X,MAP_Y)
          
      

    mc = 0
    for row in minefield:
      for cell in row:
        mc += cell

    if mc == MINE_COUNT:
      verif_count += 1

  print(f'{verif_count}/{TEST_COUNT} valid mine placement')
  for statistic_name, statistic_object in statistics.items():
    statistic_object.announce()

#example_usage()


#UI CODE BELOW HERE
#################################################################
##################################################################
#################################################################
##UI CODE BELOW HERE

create_window = None
create_warning_window = None
simulation_result_window = None
running_all_simulations = False

root = tk.Tk()
root.title("Minesweeper Simulator")
root.geometry("1000x1000")

root.grid_columnconfigure(0,weight=1)
root.grid_columnconfigure(1,weight=1)

class ProgressList(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.rows = {}      # internal storage: id → widgets
        self._counter = 0   # auto-increment row ID
        self.config(bg="white", height=200, width=400)
        self.columnconfigure(1, weight=1)

    def add(self, text):
        """Create a new row with automatically assigned ID."""
        row_id = f"row_{self._counter}"
        self._counter += 1

        # Row UI container
        row_frame = tk.Frame(self)
        row_frame.pack(fill="x", pady=2)

        # Label
        label = tk.Label(row_frame, text=text, anchor="w")
        label.pack(side="left")

        # Progress bar
        bar = ttk.Progressbar(row_frame, orient="horizontal", mode="determinate")
        bar.pack(side="left", fill="x", expand=True, padx=5)

        # Remove button
        remove_btn = tk.Button(row_frame, text="X", command=lambda: self.clean_remove(row_id))
        remove_btn.pack(side="right")

        # Store widget references
        self.rows[row_id] = {
            "frame": row_frame,
            "label": label,
            "bar": bar
        }

        return row_id  # <-- return ID to caller
    def clean_remove(self, row_id):
      self.remove(row_id)
      simulations_by_row_id[row_id].running = False
      populate_treeview()
    def update_list(self, row_id, value, simulation):
        """Update the progress bar."""
        if row_id in self.rows:
            self.rows[row_id]["bar"]["value"] = value
            self.rows[row_id]["label"]["text"] = f"Sim: {simulation.rounds} rounds, {simulation.width}x{simulation.height}, {simulation.mine_density}% mines, {simulation.completion_percentage:.3f}% complete, eta: {time.strftime('%H:%M:%S', time.gmtime(((time.time() - simulation.start_time)/simulation.completion_percentage*100)- (time.time() - simulation.start_time)))}s"
    def remove(self, row_id):
        """Remove a row cleanly."""
        if row_id in self.rows:
            self.rows[row_id]["frame"].destroy()
            del self.rows[row_id]

tk.Label(root, text="Simulations In Progress:").grid(row=8, column=0, columnspan=2, padx=10, pady=(10,0))
progress_list = ProgressList(root)
progress_list.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)


class round: 
  def __init__(self, start_pos: tuple[int,int], mine_positions: list[tuple[int,int]], simulation) -> None:
    self.start_position: tuple[int,int] = start_pos
    self.mine_positions: list[tuple[int,int]] = mine_positions
    self.tags: set[completion_tags] = set()
    self.simulation: minesweeper_simulation = simulation
    self.changes: List[BoardChange] = [BoardChange(start_pos,MAPCOLORS[0],"0","Starting Position")]


class minesweeper_simulation:
  def __init__(self, 
               rounds : int, 
               width : int, 
               height : int,
               mine_density : float
               ) -> None:
    self.rounds = rounds
    self.width = width
    self.height = height
    self.mine_density = mine_density
    self.mine_count = -1
    self.statistics = deepcopy(statistics)
    self.treeview_item_id : Optional[str] = None
    self.running = False
    self.completion_percentage = 0.0
    self.start_time = None

    self.round_list: list[round] = []

    self.UPDATE_DB = 0.5
  def get_mine_count(self):
    return int(self.width*self.height*self.mine_density/100)
  def solve_and_validate_mine_count(self):
    mc = self.get_mine_count()
    if mc >= self.height*self.width-9:
      return False
    self.mine_count = mc
    return True
  
  def run(self):
    if self.solve_and_validate_mine_count():
      last_update_time = 0
      self.start_time = time.time()
      self.running = True

      for i in range(self.rounds):
        if not self.running:
          return
        minefield, start_position_x, start_position_y,mine_positions = generate_minefield(self.width, self.height, self.mine_count)
        this_round: round = round((start_position_x,start_position_y),mine_positions, self)
        score_map,dico,final_probability_by_cell,probability_frequencies, tags,solved_cell_set, changes = solve_minefield(minefield, self.width, self.height, start_position_x, start_position_y,mine_positions)
        this_round.tags = tags
        this_round.changes = changes
        self.round_list.append(this_round)
        valid, tags = verify_board(score_map, minefield, tags)
        if completion_tags.COMPLETE in tags:
          self.statistics[completion_tags.COMPLETE.name].count += 1
        for tag in tags:
          self.statistics[tag.name].count += 1
      
        if self.treeview_item_id is not None and (time.time() - last_update_time) > self.UPDATE_DB:
          progress = int((i+1)/self.rounds*100)
          self.completion_percentage = (i+1)/self.rounds*100
          root.after(0, progress_list.update_list, self.treeview_item_id, progress, self)
          last_update_time = time.time()

      if self.treeview_item_id is not None:
        root.after(0, progress_list.update_list, self.treeview_item_id, 100, self)
        root.after(0, progress_list.remove, self.treeview_item_id)
      move_simulation_to_processed(self)
      
  def register_updater(self, treeview_item_id):
    self.treeview_item_id = treeview_item_id

  def save(self, path):
    ''' 
    Layout

    coordinateByteLength 1b
    changeCountByteLength 4b
    stringMapIndexByteLength 1b
    
    Width 4b
    Height 4b
    rounds 4b
    mineDensity 4b

    string map:
    stringCount 1b
    per string:
    StringByteLength 2b
    stringbytes...

    statmap:
    statCount 1b
    statByteWidth 1b
    per stat:
    statByteLength 2b
    statStringBytes...

    per round:
    ChangeCount ChangeCountByteLength
    StartX coordinateByteLength
    StartY coordinateByteLength
    enumBitmask statByteWidth
    

    per Change:
    ChangeCoordinateX coordinateByteLength
    ChangeCoordinateY coordinateByteLength
    Color 1b (0-10 of the presets)
    cellText stringMapIndexByteLength
    changeText stringMapIndexByteLength

    
    '''

    #solve the variable lengths:
    stringDict: dict[str, int] = {}
    reverseStringList: list[str] = []
    lowest_free_string_index = 0

    highest_coordinate = 0
    highest_change_len = 0

    enmDict: dict[str,int] = {}
    reverseEnumList: list[str] = []
    lowest_free_enum_index = 0

    for enm in completion_tags:
      if enm.name not in enmDict:
        enmDict[enm.name] = lowest_free_enum_index
        reverseEnumList.append(enm.name)
        lowest_free_enum_index += 1

    for round in self.round_list:
      highest_change_len = max(highest_change_len,len(round.changes))
      for change in round.changes:
        if change.cell_text not in stringDict:
          stringDict[change.cell_text] = lowest_free_string_index
          print(f"STRING SAVED:'{change.cell_text}'")
          reverseStringList.append(change.cell_text)
          lowest_free_string_index+=1
        if change.change_text not in stringDict:
          stringDict[change.change_text] = lowest_free_string_index
          print(f"STRING SAVED:'{change.change_text}'")
          reverseStringList.append(change.change_text)
          lowest_free_string_index+=1
        highest_coordinate = max(highest_coordinate,change.coordinate[0],change.coordinate[1])
    print("string save end")
    
    #compute variable byte length based on maximums in dataset
    coordinateByteLength = (highest_coordinate.bit_length() - 1) // 8 + 1
    changeByteLength = (highest_change_len.bit_length() - 1) // 8 + 1
    stringMapIndexByteLength = ((lowest_free_string_index - 1).bit_length() - 1) // 8 + 1


    #fill bytearray
    buffer = bytearray()

    #magic
    buffer += SAVEFILEMAGIC

    #metadata
    buffer += struct.pack("<B",coordinateByteLength)
    buffer += struct.pack("<I",changeByteLength)
    buffer += struct.pack("<B",stringMapIndexByteLength)

    #sim data
    buffer += struct.pack("<I",self.width)
    buffer += struct.pack("<I",self.height)
    buffer += struct.pack("<I",self.rounds)
    buffer += struct.pack("<f",self.mine_density)
    
    #string map
    buffer += struct.pack("<B",lowest_free_string_index)
    for string in reverseStringList:
      data = string.encode()
      buffer += struct.pack("<H",len(data))
      buffer += data
    print(f"LOW:{lowest_free_string_index}, LEN:{len(stringDict)}")

    #stat map
    statByteWidth = lowest_free_enum_index // 8 + 1
    buffer += struct.pack("<B", lowest_free_enum_index)
    buffer += struct.pack("<B", statByteWidth)
    for enm in reverseEnumList:
      data = enm.encode()
      buffer += struct.pack("<H", len(data))
      buffer += data
    #map data
    for round in self.round_list:
      #start postiiion and change count
      buffer += len(round.changes).to_bytes(changeByteLength,byteorder="little")
      buffer += round.start_position[0].to_bytes(coordinateByteLength,byteorder="little")
      buffer += round.start_position[1].to_bytes(coordinateByteLength,byteorder="little")
      bitmask = 0
      for enm in round.tags:
        bitmask |= 1 << enmDict[enm.name]
      print("EEK!")
      print(statByteWidth)
      print(lowest_free_enum_index)
      print(len(reverseEnumList))
      print(bitmask.bit_length())
      print(statByteWidth*8)
      buffer += bitmask.to_bytes(statByteWidth, byteorder="little")
      #per change data
      for change in round.changes:
        #change coord
        buffer += change.coordinate[0].to_bytes(coordinateByteLength,byteorder="little")
        buffer += change.coordinate[1].to_bytes(coordinateByteLength,byteorder="little")
        #change color
        buffer += struct.pack("<B",REVERSEMAPCOLORS[change.color])
        #change strmap index
        buffer += stringDict[change.cell_text].to_bytes(stringMapIndexByteLength,byteorder="little")
        buffer += stringDict[change.change_text].to_bytes(stringMapIndexByteLength,byteorder="little")

    compressed = zlib.compress(buffer,level=9)

    #write compressed to file....
    with open(path, "wb") as f:
      f.write(compressed)
    with open("DUPEUNCOM.mswpssf", "wb") as f:
      f.write(buffer)
    print("save end")

  def load(self, path):
    with open(path, "rb") as f:
      compressed = f.read()
    
    buffer = zlib.decompress(compressed)
    offset = 0

    magic = buffer[0:7]
    offset += 7
    if magic != SAVEFILEMAGIC:
      print("AHHH MAGIC MISMATCH")
      assert(False)
      return
    
    coordinateByteLength,changeByteLength,stringMapIndexByteLength = struct.unpack_from("<BIB", buffer, offset)
    offset += struct.calcsize("B")
    offset += struct.calcsize("I")
    offset += struct.calcsize("B")

    Width, = struct.unpack_from("<I", buffer, offset)
    offset += struct.calcsize("I")
    Height, = struct.unpack_from("<I", buffer, offset)
    offset += struct.calcsize("I")
    Rounds, = struct.unpack_from("<I", buffer, offset)
    offset += struct.calcsize("I")
    MineDensity, = struct.unpack_from("<f", buffer, offset)
    offset += struct.calcsize("f")
    stringCount, = struct.unpack_from("<B", buffer, offset)
    offset += struct.calcsize("B")

    self.width = Width
    self.height = Height
    self.rounds = Rounds
    self.mine_density = MineDensity

    stringList: List[str] = []

    stringIndex = 0
    print(f"sIndex:{stringCount}")
    while stringIndex < stringCount:
      stringByteLength = struct.unpack_from("<H", buffer, offset)[0]
      offset += struct.calcsize("H")
      data = buffer[offset:offset+stringByteLength].decode()
      offset += stringByteLength
      print(stringByteLength)
      print(data)
      stringList.append(data)
      stringIndex += 1

    statCount, = struct.unpack_from("<B", buffer, offset)
    offset += struct.calcsize("B")
    statByteWidth, = struct.unpack_from("<B", buffer, offset)
    offset += struct.calcsize("B")
    enmList: List[str] = []
    enmIndex = 0
    while enmIndex < statCount:
      enmLength, = struct.unpack_from("<H", buffer, offset)
      offset += struct.calcsize("H")
      data = buffer[offset:offset+enmLength].decode()
      print("A")
      offset += enmLength
      enmList.append(data)
      enmIndex += 1
    print("B")
    roundIndex = 0
    self.round_list.clear()
    while roundIndex < Rounds:
      changeCount = int.from_bytes(buffer[offset:offset+changeByteLength], "little")
      offset += changeByteLength
      StartX = int.from_bytes(buffer[offset:offset+coordinateByteLength], "little")
      offset += coordinateByteLength
      StartY = int.from_bytes(buffer[offset:offset+coordinateByteLength], "little")
      offset += coordinateByteLength
      thisRound = round((StartX,StartY),[(0,0)],self)
      data = int.from_bytes(buffer[offset:offset+statByteWidth], "little")
      offset += statByteWidth
      for i in range(min(statCount,len(enmList))):
        if (data >> i) & 1:
          tag = completion_tags.__members__.get(enmList[i])
          if tag is not None:
            thisRound.tags.add(tag)
      self.round_list.append(thisRound)
      changeIndex = 0
      while changeIndex < changeCount:
        ChangeCoordinateX = int.from_bytes(buffer[offset:offset+coordinateByteLength], "little")
        offset += coordinateByteLength
        ChangeCoordinateY = int.from_bytes(buffer[offset:offset+coordinateByteLength], "little")
        offset += coordinateByteLength
        color = struct.unpack_from("<B",buffer,offset)[0]
        offset += struct.calcsize("B")
        cellText = stringList[int.from_bytes(buffer[offset:offset+stringMapIndexByteLength], "little")]
        offset += stringMapIndexByteLength
        changeText = stringList[int.from_bytes(buffer[offset:offset+stringMapIndexByteLength], "little")]
        offset += stringMapIndexByteLength
        thisRound.changes.append(BoardChange((ChangeCoordinateX,ChangeCoordinateY), MAPCOLORS[color],cellText,changeText))
        changeIndex += 1
      roundIndex += 1

    self.mine_count = self.get_mine_count()

    print(f"coordByteLength:{coordinateByteLength}")
    print(f"changeByteLength:{changeByteLength}")
    print(f"stringMapIndexByteLength:{stringMapIndexByteLength}")
    print(f"Width:{Width}")
    print(f"Height:{Height}")
    print(f"Rounds:{Rounds}")
    print(f"MineDensity:{MineDensity}")
    print(f"stringCount:{stringCount}")



    


simulations_by_row_id : dict[str, minesweeper_simulation] = {}



simulation_backlog : list[minesweeper_simulation] = []
processed_simulation_backlog : list[minesweeper_simulation] = []

def warn_create(warnings :list[str]):
  global create_warning_window
  if create_warning_window and create_warning_window.winfo_exists():
    create_warning_window.lift()
  else:
    create_warning_window = tk.Toplevel(root)
    create_warning_window.title("Create Simulation Warnings")
    create_warning_window.geometry(f"400x{len(warnings)*30}")

  create_warning_window.grid_columnconfigure(0,weight=1)
  index = 0
  for warning in warnings:
    tk.Label(create_warning_window,text=warning).grid(row=index,column=0)
    create_warning_window.grid_rowconfigure(index,weight=1)

    index += 1

def create_simulation_window():
  global create_window
  if create_window and create_window.winfo_exists():
    create_window.lift()
    return
  
  create_window = tk.Toplevel(root)
  create_window.title("Create simulation")
  create_window.geometry("350x250")

  create_window.grid_columnconfigure(0,weight=1)
  create_window.grid_columnconfigure(1,weight=1)
  create_window.grid_rowconfigure(0,weight=1)
  create_window.grid_rowconfigure(1,weight=1)
  create_window.grid_rowconfigure(2,weight=1)
  create_window.grid_rowconfigure(3,weight=1)
  create_window.grid_rowconfigure(4,weight=2)


  simulation = minesweeper_simulation(0, 0, 0, 0)
  # ----- Row 0 -----
  tk.Label(create_window, text="Number of games to simulate:").grid(row=0, column=0,sticky="e")
  entry_rounds = tk.Entry(create_window)
  entry_rounds.grid(row=0, column=1)

  # ----- Row 1 -----
  tk.Label(create_window, text="Board width:").grid(row=1, column=0,sticky="e")
  entry_width = tk.Entry(create_window)
  entry_width.grid(row=1, column=1)

  # ----- Row 2 -----
  tk.Label(create_window, text="Board height:").grid(row=2, column=0,sticky="e")
  entry_height = tk.Entry(create_window)
  entry_height.grid(row=2, column=1)

  # ----- Row 3 -----
  tk.Label(create_window, text="Mine density (%):").grid(row=3, column=0,sticky="e")
  entry_density = tk.Entry(create_window)
  entry_density.grid(row=3, column=1)

  def verify():
    nonlocal entry_density
    nonlocal entry_height
    nonlocal entry_width
    nonlocal entry_rounds
    nonlocal simulation
  
    warnings = []
    try:
      simulation.height = int(entry_height.get())
    except:
      warnings.append("Simulation height not an integer")

    try:
      simulation.width = int(entry_width.get())
    except:
      warnings.append("Simulation width not an integer")

    try:
      simulation.rounds = int(entry_rounds.get())
    except:
      warnings.append("Simulation rounds not an integer")

    try:
      simulation.mine_density = int(entry_density.get())
    except:
      warnings.append("Simulation density not a number")

    if len(warnings) == 0:
      if simulation.height < 1:
        warnings.append("Simulation height less than 1")
      if simulation.width < 1:
        warnings.append("Simulation width less than 1")
      if simulation.rounds < 1:
        warnings.append("Simulation height less than 1")
      if simulation.mine_density < 1 or simulation.mine_density > 100:
        warnings.append("Simulation mine density out of bounds (1 to 100)")

      if not simulation.solve_and_validate_mine_count():
        warnings.append("Simulation mine density too high for a starting location")
    if len(warnings) != 0:
      warn_create(warnings)
    else:
      simulation_backlog.append(simulation)
      populate_treeview()
      #show success before closing
      messagebox.showinfo("Success", "Simulation created successfully.")
      create_window.destroy() # type: ignore
    


  # ----- Button -----
  bt = tk.Button(create_window, text="Start Simulation", command=verify)\
      .grid(row=4, column=0, columnspan=2, pady=10)

create_simulation = tk.Button(root, text="Create simulation", command=create_simulation_window)
create_simulation.grid(row=0,column=0, columnspan=2)
tk.Label(root, text="Simulation backlog:").grid(row=1,column=0, columnspan=2)
tree = ttk.Treeview(root, columns=("rounds", "width", "height", "density", "mines"), show="headings")
tree.column("rounds", width=100)
tree.column("width", width=100)
tree.column("height", width=100)
tree.column("density", width=100)
tree.column("mines", width=100)

tree.heading("rounds", text="Rounds")
tree.heading("width", text="Width")
tree.heading("height", text="Height")
tree.heading("density", text="Density")
tree.heading("mines", text="Mines")
tree.grid(row=2,column=0, columnspan=2)

tree_table = {}

tk.Label(root, text="Processed simulations:").grid(row=5,column=0, columnspan=2)
processed_simulations_tree = ttk.Treeview(root, columns=("rounds", "width", "height", "density", "mines"), show="headings")
processed_simulations_tree.column("rounds", width=100)
processed_simulations_tree.column("width", width=100)
processed_simulations_tree.column("height", width=100)
processed_simulations_tree.column("density", width=100)
processed_simulations_tree.column("mines", width=100)

processed_simulations_tree.heading("rounds", text="Rounds")
processed_simulations_tree.heading("width", text="Width")
processed_simulations_tree.heading("height", text="Height")
processed_simulations_tree.heading("density", text="Density")
processed_simulations_tree.heading("mines", text="Mines")
processed_simulations_tree.grid(row=7,column=0, columnspan=2)

processed_simulations_tree_table = {}

def populate_treeview():
  tree.delete(*tree.get_children())
  for simulation in simulation_backlog:
    if simulation.running:
      continue
    tree_object = tree.insert("", "end", values=(f"{simulation.rounds}", f"{simulation.width}", f"{simulation.height}", f"{simulation.mine_density}%", f"{simulation.mine_count}"))
    tree_table[tree_object] = simulation

def start_selected_simulation():
  selected_items = tree.selection()
  if not selected_items:
    messagebox.showwarning("No Selection", "Please select a simulation to start.")
    return

  selected_item = selected_items[0]
  simulation = tree_table.get(selected_item)
  if simulation and not simulation.running:
    #run the simulation on another thread
    sim_id = progress_list.add(f"Sim: {simulation.rounds} rounds, {simulation.width}x{simulation.height}, {simulation.mine_density}% mines, {simulation.completion_percentage}% complete")
    simulation.register_updater(sim_id)
    simulations_by_row_id[sim_id] = simulation
    threading.Thread(target=simulation.run, daemon=True).start()
    messagebox.showinfo("Simulation Started", f"Started simulation with {simulation.rounds} rounds.")

    populate_treeview()
  else:
    messagebox.showerror("Error", "Selected simulation not found or is already running.")
start_simulation_button = tk.Button(root, text="Start Selected Simulation", command=start_selected_simulation)
start_simulation_button.grid(row=3,column=0, sticky="e")
delete_selected_simulation_button = tk.Button(root, text="Delete Selected Simulation", command=lambda: delete_selected_simulation(tree.selection()))
delete_selected_simulation_button.grid(row=4,column=0, columnspan=2)

def delete_selected_simulation(selected_item):
  if not selected_item:
    messagebox.showwarning("No Selection", "Please select a simulation to delete.")
    return

  for item in selected_item:
    simulation = tree_table.get(item)
    if simulation:
      simulation_backlog.remove(simulation)
  
  populate_treeview()

def move_simulation_to_processed(simulation: minesweeper_simulation):
  if simulation in simulation_backlog:
    simulation_backlog.remove(simulation)
    processed_simulation_backlog.append(simulation)
    populate_treeview()
    populate_processed_treeview()

def run_all_simulations():
  if len(simulation_backlog) < 1:
    messagebox.showwarning("No Simulations", "There are no simulations to run.")
    return
  
  for simulation in simulation_backlog:
    #messagebox.showinfo("Simulation Started", f"Started simulation with {simulation.rounds} rounds.")
    #run the simulation on another thread
    if simulation.running:
      messagebox.showinfo("Simulation Skipped", f"Will skip simulation with {simulation.rounds} rounds as it is already running, press OK to continue running simulations.")
      continue
    sim_id = progress_list.add(f"Sim: {simulation.rounds} rounds, {simulation.width}x{simulation.height}, {simulation.mine_density}% mines, {simulation.completion_percentage}% complete")
    simulation.register_updater(sim_id)
    simulations_by_row_id[sim_id] = simulation
    threading.Thread(target=simulation.run, daemon=True).start()
  messagebox.showinfo("Simulations Started", "All simulations started.")

  populate_treeview()

  


run_all_simulations_button = tk.Button(root, text="Run All Simulations", command=run_all_simulations)
run_all_simulations_button.grid(row=3,column=1, sticky="w")

def export_statistics(selected_item, is_all : bool = False):
  simulations_to_export: List[minesweeper_simulation] = []
  if is_all:
    simulations_to_export = processed_simulation_backlog
  else:
    simulations_to_export = []
    for item in selected_item:
      simulation = processed_simulations_tree_table.get(item)
      if simulation:
        simulations_to_export.append(simulation)

  if len(simulations_to_export) == 0:
    messagebox.showwarning("No Simulations Selected", "No processed simulations were selected for export.")
    return
  #export to csv
  #open a file dialog to select save location
  file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
  if not file_path:
    messagebox.showwarning("No File Selected", "No file was selected for saving.")
    return
  
  with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    #write header
    header = ["Rounds", "Width", "Height", "Mine Density", "Mine Count"]
    for tag_name, tag in statistics.items():
      header.append(tag.message)
    writer.writerow(header)

    for simulation in simulations_to_export:
      row = [simulation.rounds, simulation.width, simulation.height, simulation.mine_density, simulation.mine_count]
      for tag_name, statistic in simulation.statistics.items():
        row.append(statistic.count)
      writer.writerow(row)

    messagebox.showinfo("Export Successful", f"Simulation statistics exported successfully to {file_path}.")

def saveSimulations(selected_item, is_all:bool = False):
  simulations_to_export: List[minesweeper_simulation] = []
  if is_all:
    simulations_to_export = processed_simulation_backlog
  else:
    simulations_to_export = []
    for item in selected_item:
      simulation = processed_simulations_tree_table.get(item)
      if simulation:
        simulations_to_export.append(simulation)
  
  if len(simulations_to_export) > 1:
    folder = filedialog.askdirectory(title="Select a folder to save simulations to.")
    if folder:
      ts = time.strftime("%Y%m%d_%H%M%S") 
      for i, sim in enumerate(simulations_to_export):
        filename = f"{ts}_simulation_{i+1}.mswpsf"
        sim.save(os.path.join(folder, filename))
  else:
    file_path = filedialog.asksaveasfilename(defaultextension=".mswpsf", filetypes=[("Minesweeper Save Files", "*.mswpsf")])
    ts = time.strftime("%Y%m%d_%H%M%S") 
    for i, sim in enumerate(simulations_to_export):
      sim.save(file_path)

def loadSimulations():
  file_paths = filedialog.askopenfilenames(defaultextension=".mswpsf", filetypes=[("Minesweeper Save Files", "*.mswpsf")])
  for path in file_paths:
    sim = minesweeper_simulation(0,0,0,0)
    try:
      sim.load(path)
      simulation_backlog.append(sim)
      move_simulation_to_processed(sim)
    except Exception as e:
      print(f"Failed to load {path} with error {e}")


export_statistics_button = tk.Button(root, text="Export Selected Simulation Statistics", command=lambda: export_statistics(processed_simulations_tree.selection()))
export_statistics_button.grid(row=10,column=0, sticky="e")

export_all_statistics_button = tk.Button(root, text="Export All Simulation Statistics", command=lambda: export_statistics(None, True))
export_all_statistics_button.grid(row=10,column=1, sticky="w")

browse_selected_simulation_button = tk.Button(root, text="Browse Rounds Of Selected Simulation", command=lambda: open_round_browser(root,processed_simulations_tree.selection(), view_round))
browse_selected_simulation_button.grid(row=11,column=0, columnspan=2)

saveSelectedSimulationButton = tk.Button(root, text="Save Selected Simulation/s", command=lambda: saveSimulations(processed_simulations_tree.selection()))
saveSelectedSimulationButton.grid(row=12,column=0, sticky="e")

saveAllSimulationButton = tk.Button(root, text="Save All Simulations", command=lambda: saveSimulations(processed_simulations_tree.selection(), True))
saveAllSimulationButton.grid(row=12,column=1, sticky="w")

loadSimulationsButton = tk.Button(root, text="Load Simulations", command=lambda: loadSimulations())
loadSimulationsButton.grid(row=13,column=0, columnspan=2)

def populate_processed_treeview():
  processed_simulations_tree.delete(*processed_simulations_tree.get_children())
  for simulation in processed_simulation_backlog:
    pst = processed_simulations_tree.insert("", "end", values=(f"{simulation.rounds}", f"{simulation.width}", f"{simulation.height}", f"{simulation.mine_density}%", f"{simulation.mine_count}"))
    processed_simulations_tree_table[pst] = simulation

def view_round(round:round):
  renderer: MinesweeperRenderer = MinesweeperRenderer(round.simulation.width, round.simulation.height)
  renderer.run_round(round)

def open_round_browser(
    root: tk.Tk,
    selected,
    on_view_round: Callable[[round], None]
):
    if len(selected) != 1:
        messagebox.showinfo("View round", "Please only select one simulation to view.")
        return

    simulation = processed_simulations_tree_table.get(selected[0])
    if simulation is None:
        messagebox.showinfo("Error", "Couldn't find selected simulation somehow")
        return

    win = tk.Toplevel(root)
    win.title("Simulation Rounds")
    win.geometry("800x500")

    PAGE_SIZE = 100
    current_page = 0
    filtered_rounds: list[round] = []

    # -------------------------------
    # Tag filter panel
    # -------------------------------
    filter_frame = ttk.LabelFrame(win, text="Filter by Completion Tags")
    filter_frame.pack(fill="y", padx=8, pady=6)

    tag_vars = {}

    for tag in completion_tags:
        var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(
            filter_frame,
            text=tag.name,
            variable=var
        )
        chk.pack( padx=4)
        tag_vars[tag] = var

    # -------------------------------
    # Scrollable list setup
    # -------------------------------
    container = ttk.Frame(win)
    container.pack(fill="both", expand=True, padx=8, pady=8)

    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # -------------------------------
    # Row rendering with pagination
    # -------------------------------
    def render_rows():
        nonlocal filtered_rounds, current_page

        for child in scroll_frame.winfo_children():
            child.destroy()

        selected_tags = {tag for tag, var in tag_vars.items() if var.get()}

        filtered_rounds = [
            r for r in simulation.round_list
            if selected_tags.issubset(r.tags)
        ]

        max_pages = max(1, (len(filtered_rounds) - 1) // PAGE_SIZE + 1)
        current_page = max(0, min(current_page, max_pages - 1))

        start = current_page * PAGE_SIZE
        end = start + PAGE_SIZE

        for idx, r in enumerate(filtered_rounds[start:end], start=start):
            row = ttk.Frame(scroll_frame)
            row.pack(fill="x", pady=2)

            ttk.Label(
                row,
                text=f"Round {idx}",
                width=12
            ).pack(side="left", padx=4)

            tag_text = ", ".join(tag.name for tag in r.tags) or "—"
            ttk.Label(
                row,
                text=tag_text
            ).pack(side="left", fill="x", expand=True, padx=6)

            ttk.Button(
                row,
                text="View / Step",
                command=lambda r=r: on_view_round(r)
            ).pack(side="right", padx=4)

        page_label.config(
            text=f"Page {current_page + 1} / {max_pages} ({len(filtered_rounds)} rounds)"
        )

        canvas.yview_moveto(0)

    # -------------------------------
    # Pagination controls
    # -------------------------------
    nav_frame = ttk.Frame(win)
    nav_frame.pack(fill="x", padx=8, pady=4)

    page_label = ttk.Label(nav_frame, text="")
    page_label.pack(side="left")

    def prev_page():
        nonlocal current_page
        if current_page > 0:
            current_page -= 1
            render_rows()

    def next_page():
        nonlocal current_page
        max_pages = max(1, (len(filtered_rounds) - 1) // PAGE_SIZE + 1)
        if current_page < max_pages - 1:
            current_page += 1
            render_rows()

    ttk.Button(nav_frame, text="◀ Prev", command=prev_page).pack(side="right", padx=4)
    ttk.Button(nav_frame, text="Next ▶", command=next_page).pack(side="right")

    # -------------------------------
    # Apply filter button
    # -------------------------------
    def apply_filter():
        nonlocal current_page
        current_page = 0
        render_rows()

    ttk.Button(
        filter_frame,
        text="Apply Filter",
        command=apply_filter
    ).pack(side="right", padx=6)

    render_rows()



root.mainloop()