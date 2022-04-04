import numpy as np
from random import sample, random

size = 9  # sudoku size x size

def changeColumns(matrix):
  columns = []
  for i in range(size):
    columns.append([])
    for j in range(size):
      columns[i].append(matrix[j][i])
  return columns[:]

class Candidate(object):
  def __init__(self, template):
    self.template = template
    self.value = template.copy()
    self.aptitude = 0

  def __str__(self):
    candidate_str = 'sudoku:\n'
    for row in self.value:
      for value in row:
        candidate_str += str(value) + ' '
      candidate_str += '\n'
    candidate_str += 'aptitude: ' + str(self.aptitude)
    return candidate_str

  def fitness(self):
    score = 0
    # Checks columns
    columns = changeColumns(self.value)
    
    for column in columns:
      score += len({i: column.count(i) for i in column})
    self.aptitude = score/size**2

  def mutate(self):
    for i in range(size):
      # Gets all values that cant be moved
      locked = self.template[i].copy()
      try:
        while 1:
          locked.remove(0)
      except: pass

      change = []
      for j in range(1, size+1):
        if j not in locked: change.append(j)
      if (len(change) < 2): continue

      # Mutates the row
      a, b = np.random.permutation(change)[:2]
      a_index = list(self.value[i]).index(a)
      b_index = list(self.value[i]).index(b)
      self.value[i][a_index] = b
      self.value[i][b_index] = a
    self.fitness()

class Population(object):
  def __init__(self, template):
    self.template = template
    self.candidates = []

  # Generates first population
  def generate(self, amount):
      # Fill the candidate
      objs = [Candidate(self.template.copy()) for _ in range(amount)]
      for obj in objs:
        for row in obj.value:
          # Remove nums duplicated in row
          nums = [x for x in range(1, size+1)]
          for value in row:
            try: nums.remove(value)
            except: pass
          
          randomized = np.random.permutation(nums)
          # Generate candidate
          for i in range(size):
            if (row[i] != 0): continue
            row[i] = randomized[0]
            randomized = np.delete(randomized, 0)
        obj.fitness()
        self.candidates.append(obj)

  def sort(self):
    self.candidates = sorted(self.candidates, key=lambda x: x.aptitude)

  def cross(self, a, b):
    c = Candidate(a.value.copy())
    numbers = [x for x in range(1, size+1)]

    for i in range(size):
      immovable = []
      replacing = []
      # Gets values that cannot be moved by template
      for value in self.template[i]:
        if (value != 0): immovable.append(value)
      
      # Gets values that cannot be moved by first parent
      while (1):
        if (len(immovable) > 5): break
        number = sample(numbers, 1)[0]
        if (number not in immovable): immovable.append(number)

      # Gets values that will be replaced
      for j in range(size):
        if (b.value[i][j] not in immovable): replacing.append(b.value[i][j])

      # Replace child
      rreplacing = np.array(replacing)
      for k in range(size):
        if (c.value[i][k] in immovable): continue
        c.value[i][k] = rreplacing[0]
        rreplacing = np.delete(rreplacing, 0)
      
    c.fitness()
    self.candidates.append(c)

class Sudoku(object):
  def __init__(self, sudoku_path):
    with open(sudoku_path, "r") as f:
        self.value = np.loadtxt(f).reshape((size, size)).astype(int)
  
  def solution(self, pop_num=100, iterations=100, select=0.5, mutate=0.1):
    population = Population(self.value.copy())
    population.generate(pop_num)
    population.sort()
    print('Generation: 0 Aptitude:', population.candidates[-1].aptitude)
    population.candidates = population.candidates[int(pop_num*select):]
    best = population.candidates[-1].aptitude
    for i in range(1, iterations+1):
      while (len(population.candidates) < pop_num):
        candidates = sample(population.candidates, 2)
        population.cross(candidates[0], candidates[1])

      # Tries to mutate children
      for candidate in population.candidates:
        if (random() < mutate): candidate.mutate()
      population.sort()

      if (population.candidates[-1].aptitude == 1):
        print('sudoku solved in generation:', i, '\n', population.candidates[-1])
        return
      
      if (best < population.candidates[-1].aptitude):
        print('Generation:', i, 'Aptitude:', population.candidates[-1].aptitude)
        best = population.candidates[-1].aptitude
      population.candidates = population.candidates[int(pop_num*select):]
    print('solution not found in', i, 'but got', population.candidates[-1])

s = Sudoku('puzzle_mild.txt')
s.solution(pop_num=300, iterations=1000, select=0.3, mutate=0.45)
