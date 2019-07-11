import random

class Matrix:
    def __init__(self, arg):
        assert all(type(i) is list and len(arg[0]) == len(i) for i in arg)
        self._matrix = arg
        self._rows = len(arg)
        self._cols = len(arg[0])

    def __str__(self):
        return "[" + "\n ".join([",".join(str(i) for i in r) for r in self._matrix]) + "]"

    def __repr__(self):
        return "Matrix({})".format(self._matrix)

    def getRows(self):
        return self._rows

    def getCols(self):
        return self._cols

    #returns a list of the index row in matrix
    def row(self, index):
        return self._matrix[index].copy()

    #returns a list of the column
    def col(self, index):
        return [r[index] for r in self._matrix]
    
    def toList(self):
        if self.getRows() == 1: #is it a vector
            return self._matrix[0]
        return self._matrix

    def __getitem__(self, index):
        if type(index) is tuple:
            return self._matrix[index[0]][index[1]]
        else:
            raise IndexError()

    def __setitem__(self, index, value):
        if type(index) is tuple:
            self._matrix[index[0]][index[1]] = value
        else:
            raise IndexError()

    def randomize(self):
        for r in range(self.getRows()):
            for c in range(self.getCols()):
                #random float between -1 and 1
                self._matrix[r][c] = random.random() * 2 - 1

    def elementmul(self, other):
        if type(other) is Matrix:
            assert self.getRows() == other.getRows() and self.getCols() == other.getCols(),"dimensions unequal"
            m = Matrix([l.copy() for l in self._matrix])
            for r in range(len(self._matrix)):
                for c in range(len(self._matrix[r])):
                    m._matrix[r][c] *= other._matrix[r][c]
            return m
        else:
            raise TypeError()

    def elementmap(self, f):
        m = Matrix([l.copy() for l in self._matrix])
        for r in range(len(self._matrix)):
            for c in range(len(self._matrix[r])):
                m._matrix[r][c] = f(self._matrix[r][c])
        return m
    
    def __add__(self, other):
        # scalar addition
        if type(other) in (int, float):
            m = Matrix([l.copy() for l in self._matrix])
            for r in range(len(self._matrix)):
                for c in range(len(self._matrix[r])):
                    m._matrix[r][c] += other
            return m
        # matrix addition
        elif type(other) is Matrix:
            assert self.getRows() == other.getRows() and self.getCols() == other.getCols(),"dimensions unequal"
            m = Matrix([l.copy() for l in self._matrix])
            for r in range(len(self._matrix)):
                for c in range(len(self._matrix[r])):
                    m._matrix[r][c] += other._matrix[r][c]
            return m
        else:
            return NotImplemented

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        #scalar multiplication
        if type(other) in (int,float):
            m = Matrix([l.copy() for l in self._matrix])
            for r in range(self.getRows()):
                for c in range(self.getCols()):
                    m._matrix[r][c] *= other
            return m
        #dot product
        elif type(other) is Matrix:
            assert self.getCols() == other.getRows(), "dimensions rows and cols unequal"
            m = Matrix([[0 for c in range(other.getCols())] for r in range(self.getRows())])
            for r in range(self.getRows()):
                for c in range(other.getCols()):
                    value = 0
                    for i in range(self.getCols()):
                        value += self[r, i] * other[i, c]
                    m._matrix[r][c] = value
            return m              
        else:
            return NotImplemented

    @staticmethod
    def transpose(mat):
        m = Matrix([[0 for _ in range(mat.getRows())] for _ in range(mat.getCols())])
        for r in range(mat.getRows()):
            for c in range(mat.getCols()):
                m._matrix[c][r] = mat._matrix[r][c]
        return m

    @staticmethod
    def zeros(rows, cols):
        return Matrix([[0 for i in range(cols)] for i in range(rows)])

    @staticmethod
    def vector(array):
        return Matrix([[i] for i in array])

