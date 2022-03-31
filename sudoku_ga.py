import numpy
import random
random.seed()

digits = 9  # Number of digits.

class Population(object):
    #A set of candidate solutions (the population chromosomes) to the Sudoku puzzle.
    def __init__(self):
        self.candidates = []
        return

    def seed(self, numCandidates, given):
        self.candidates = []
        
        # Determine the possible  values that each square can take.
        assistant = Candidate()
        assistant.values = [[[] for j in range(0, digits)] for i in range(0, digits)]
        for row in range(0, digits):
            for column in range(0, digits):
                for value in range(1, 10):
                    if((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        assistant.values[row][column].append(value)
                    elif(given.values[row][column] != 0):
                        # Given/known value from file.
                        assistant.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population.       
        for p in range(0, numCandidates):
            g = Candidate()
            for i in range(0, digits): # New row in candidate.
                row = numpy.zeros(digits)

                for j in range(0, digits): # New column j value in row i.
                
                    # if the value is given, it does not change
                    if(given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    # Fill in the gaps using the assistant board.
                    elif(given.values[i][j] == 0):
                        row[j] = assistant.values[i][j][random.randint(0, len(assistant.values[i][j])-1)]

                # If we don't have a valid board, then try again.
                while(len(list(set(row))) != digits):
                    for j in range(0, digits):
                        if(given.values[i][j] == 0):
                            row[j] = assistant.values[i][j][random.randint(0, len(assistant.values[i][j])-1)]

                g.values[i] = row

            self.candidates.append(g)
        
        # Compute the aptitude of all candidates in the population.
        self.update_aptitude()
        
        print("Seeding complete.")

    def update_aptitude(self):
        # Update aptitude of every candidate
        for candidate in self.candidates:
            candidate.update_aptitude()

    def sort(self):
        # Sort the population based on aptitude.
        self.candidates = sorted(self.candidates, key=lambda x: x.aptitude)

    def sort_aptitude(self, x, y):
        # The sorting function.
        if(x.aptitude < y.aptitude):
            return 1
        elif(x.aptitude == y.aptitude):
            return 0
        else:
            return -1


class Candidate(object):
    # A candidate solutions to the Sudoku puzzle. 
    def __init__(self):
        self.values = numpy.zeros((digits, digits), dtype=int)
        self.aptitude = None
        return

    def mutate(self, mutation_rate, given):
        # Mute a un candidato eligiendo una fila, para elegir dos valores dentro de esa fila para intercambiar.

        r = random.uniform(0, 1.1)
        while(r > 1): 
            r = random.uniform(0, 1.1)
    
        success = False
        if (r < mutation_rate):  
            while(not success):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1
                
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while(from_column == to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)   

                # Check if the two places are free
                if(given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    if(not given.is_column_duplicate(to_column, self.values[row1][from_column])
                       and not given.is_column_duplicate(from_column, self.values[row2][to_column])
                       and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column])
                       and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):
                    
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True
    
        return success

    def update_aptitude(self):
        # The actual solution is defined as a 9x9 grid. If there are any duplicates then the aptitude will be lower.
        
        row_count = numpy.zeros(digits)
        column_count = numpy.zeros(digits)
        block_count = numpy.zeros(digits)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        for i in range(0, digits):  # For each column
            for j in range(0, digits):
                column_count[self.values[j][i]-1] += 1 

            column_sum += (1.0 / len(set(column_count)))/digits
            column_count = numpy.zeros(digits)

        for i in range(0, digits):  # For each row
            for j in range(0, digits): 
                row_count[self.values[i][j]-1] += 1 

            row_sum += (1.0/len(set(row_count)))/digits
            row_count = numpy.zeros(digits)

        # For each block
        for i in range(0, digits, 3):
            for j in range(0, digits, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1.0/len(set(block_count)))/digits
                block_count = numpy.zeros(digits)

        # Calculate overall aptitude
        if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
            aptitude = 1.0
        else:
            aptitude = column_sum * block_sum
        
        self.aptitude = aptitude
        
    

class Tournament(object):
    # Two individuals are selected from the population pool and a random number in [0, 1] is chosen.If this number is less 
    # than the 'selection rate' then the fitter individual is selected; otherwise, the weaker one is selected

    def __init__(self):
        return
        
    def compete(self, candidates):
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.aptitude
        f2 = c2.aptitude

        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1):  
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest

class Given(Candidate):
    # The grid containing the known values.

    def __init__(self, values):
        self.values = values
        return
        
    def is_column_duplicate(self, column, value):
        for row in range(0, digits):
            if(self.values[row][column] == value):
               return True
        return False

    def is_row_duplicate(self, row, value):
        for column in range(0, digits):
            if(self.values[row][column] == value):
               return True
        return False


    def is_block_duplicate(self, row, column, value):
        i = 3*(int(row/3))
        j = 3*(int(column/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
            return True
        else:
            return False

class CycleCrossover(object):
    #  each parent candidate mixing together in the hopes of creating a fitter child candidate

    def __init__(self):
        pass
    
    def crossover(self, parent1, parent2, crossover_rate):
        child1 = Candidate()
        child2 = Candidate()
        
        child1.values = numpy.copy(parent1.values)
        child2.values = numpy.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while(r > 1):  
            r = random.uniform(0, 1.1)
            
        if (r < crossover_rate):
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while(crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)
                
            if(crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
                
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1 = numpy.zeros(digits)
        child_row2 = numpy.zeros(digits)

        remaining = [x for x in range(1, digits+1)]
        cycle = 0
        
        while((0 in child_row1) and (0 in child_row2)): 
            if(cycle % 2 == 0):  
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]
                
                while(next != start):  
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]
                
                while(next != start):  
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]
                    
                cycle += 1

        return child_row1, child_row2  

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if(parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if(parent_row[i] == value):
                return i


class Sudoku(object):
    # Solves a given Sudoku puzzle using a genetic algorithm. 

    def __init__(self, sudoku_path):
        with open(sudoku_path, "r") as f:
            values = numpy.loadtxt(f).reshape((digits, digits)).astype(int)
            self.given = Given(values)

    def save(self, path, solution):
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(digits*digits), fmt='%d')
        return
        
    def solve(self):
        numCandidates = 1000  
        Ne = int(0.05*numCandidates) 
        Ng = 1000  
        Nm = 0  
        
        phi = 0
        sigma = 1
        mutation_rate = 0.06
    
        # Create an initial population.
        self.population = Population()
        self.population.seed(numCandidates, self.given)
    
        # For up to 10000 ageGroups...
        obsolete = 0
        for ageGroup in range(0, Ng):
        
            print("ageGroup %d" % ageGroup)
            
            # Check for a solution.
            best_aptitude = 0.0
            for c in range(0, numCandidates):
                aptitude = self.population.candidates[c].aptitude
                if(aptitude == 1):
                    print("Solution found at age group %d!" % ageGroup)
                    print(self.population.candidates[c].values)
                    return self.population.candidates[c]

                # Find the best aptitude.
                if(aptitude > best_aptitude):
                    best_aptitude = aptitude

            print("Best aptitude: %f" % best_aptitude)

            # Create the next population.
            next_population = []

            # Select the fittest candidates and preserve them for the next ageGroup.
            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Create the rest of the candidates.
            for count in range(Ne, numCandidates, 2):
              
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)
                
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)
                
                success = child1.mutate(mutation_rate, self.given)
                child1.update_aptitude()
                if(success):
                    Nm += 1
                    phi += 1
                
                success = child2.mutate(mutation_rate, self.given)
                child2.update_aptitude()
                if(success):
                    Nm += 1
                    phi += 1
                
                next_population.append(child1)
                next_population.append(child2)

           
            for e in range(0, Ne):
                next_population.append(elites[e])
                
            
            self.population.candidates = next_population
            self.population.update_aptitude()
            
           
            if(Nm == 0):
                phi = 0  
            else:
                phi = phi / Nm
            
            if(phi > 0.2):
                sigma = sigma/0.998
            elif(phi < 0.2):
                sigma = sigma*0.998

            mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
            Nm = 0
            phi = 0

            self.population.sort()
            if(self.population.candidates[0].aptitude != self.population.candidates[1].aptitude):
                obsolete = 0
            else:
                obsolete += 1

           
            if(obsolete >= 100):
                print("The population has gone obsolete. seeding again...")
                self.population.seed(numCandidates, self.given)
                obsolete = 0
                sigma = 1
                phi = 0
                Nm = 0
                mutation_rate = 0.06
        
        print("No solution found :(")
        return None
        
s = Sudoku('puzzle_mild.txt')
solution = s.solve()
if(solution):
    s.save("solution.txt", solution)
