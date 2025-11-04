import numpy as np
from atomic_number import atomic_number

class Molecule:
    '''A Molecule class is created by reading a standard xyz file of any molecule, the data in the file is then used 
    to create attributes for that molecule, for examples its number of atoms and each atoms cartesian position in a dictionary. 
    In this case, a Coulomb matrix is used as a descriptor'''
    def __init__(self, filename):
        '''Initialising the molecule by reading the file and extracting the atom symbols and their respective cartesian positions. 
        These two information is then use to perform multiple linear algebra operations on the molecule. Pass an xyz file to the function. 
        Lines 1 and 2 can have any content, BUT line 3 onwards must be in the form of "atom x y z" '''
        atom_data=[]
        with open(filename, 'r') as data: #open the file and store as data
            for line in data: #each line read separately, split, and added to the empty list
                atom = line.split() #splitting the line at the white spaces, this return a list which is made of the elements of the line
                #line is split into string objects
                atom_data.append(atom) #appending the empty list with the 
            atom_data=atom_data[2:] #deleting the first two elements of the list, skipping the first 2 ros of the xyz file
            self.natoms = len(atom_data) #the length of the new list, each list corresponds to one atom, so therefore this length should return
            #the number of atoms
            
            self.xyz = [] #an empty list which will be appended with a dictionary created from each list in the list of list atom_data
            #this loops over each element of the list of lists atom_data
            for row in atom_data: #each element of the list atom_data (which is also a list) is given the variable row
                ele = row[0] #for each row, the element at the 0 index is given the variable ele
                coords = np.array(row[1:], dtype=float) #each element from the 1 index to the end is extracted and assigned as a float
                #these elements are then stored in a numpy array of 1D
                mol_dict = {'element': ele, 'coordinates': coords} #each row is split into the atomic symbol and its cartesian position and put into a dictionary
                self.xyz.append(mol_dict) # the list is appended with the dictionary created from each row
    
            self.xyz = tuple(self.xyz) #the final list of dictionaries after every row has been looped over is changed to a tuple of dictionaries
        
    def rawCM(self):
        '''This function returns the raw Coulomb matrix of a molecule'''
        N = self.natoms #assigning the number of atoms a variable which will be used to create a matrix of NxN to be filled 
        initial_coulomb = np.zeros((N,N)) #fill a matrix of NxN with zeros
        identity_val = [] #an empty list that will be appended with the numbers that will fill the diagonal position
        for i in self.xyz:
            element = i['element'] #look for the element symbol in the disctionary of the atom
            atom_num = atomic_number[element] # look for the same element symbol from the dictionary in the dictionary of atomic numbers and return 
            #the atomic number
            d_val = 0.5*(atom_num**2.4) #using the equation for diagonal positions of atoms interacting with itself.
            identity_val.append(d_val) #appending the empty list with each atoms interaction energy, in the order of the xyz tuple dictionary, hence in
            #the order of the xyz file
        np.fill_diagonal(initial_coulomb,identity_val) #the fill_diagonal function fill the main diagonal of an array with a value, in this case
        #we are filling it with the atom interacting energies in the list identity_val.

        #this part of the code takes the matrix with the atom interacting energies and fills the off diagonal positions. 
        for i in range(N): 
            for j in range(N):
        #the nested loop above loops through the rows, i, and then the columns, j. for every one row index, it loops through every column index
                if i!=j: #when i and j are not equal, that is an off diagonal position. so the second equation for interacting energy between different
                    #atoms are used here to fill those spaces
                    atom_i = atomic_number[self.xyz[i]['element']] #searches in the xyz tuple for the atom disctionary at position i, it then brings
                    # out the value corresponding to the key 'element', the atomic symbol. This symbol is then searched as the key in atomic_number 
                    # disctionary and the value corresponding to that symbol is returned and stored in atom_i
                    atom_j = atomic_number[self.xyz[j]['element']] #same operation as above is happening here, but this time the column index number is 
                    #what begins the search.
                    vec_i = self.xyz[i]['coordinates'] #this returns the np array of xyz coordinates for the atom at tuple position i
                    vec_j = self.xyz[j]['coordinates'] #this returns the np array of xyz coordinates for the atom at tuple position j
                    matrix_fill = (atom_i*atom_j)/np.linalg.norm(vec_i-vec_j) #off_diagonal position equation is applied here norm return the norm between
                    #the vector positions for atom i and j
                    initial_coulomb[i,j] = matrix_fill #the answer to the above question is used to fill the columb at the [i][j] index position. This is
                    #done for every atoms against every other atom
        return initial_coulomb #completed coulomb is returned by the function

    
    def eigenCM(self):
        '''Returns the eigenvalues of the raw Coulomb matrix of the molecule in descending order'''
        # e_val = np.linalg.eigvalsh(self.rawCM)
        e_sort = np.sort(np.linalg.eigvalsh(self.rawCM()))[::-1]
        #eigenvalsh takes the raw matrix and returns the eigenvalues for the symmetric matrix, np.sort rearranges the values in ascending order, this
        #is reversed using [::-1]. the number of values returned is N in an NxN symmetrical matrix
        return e_sort

    def eigenCM_distance(self, mole_2):
        '''The first molecule is comapred to the second molecule, and returns the euclidean norm of the two, which is 
        their distance'''
        e_mol_1 = self.eigenCM() #the reordered eigenvalues of the self molecules
        e_mol_2 = mole_2.eigenCM() #the reordered eigenvalues of the argument molecule

        if len(e_mol_1) > len(e_mol_2): #this is comparing the sizes of the molecules. as N is the number of atoms and N numbers is returned eigenvalsh,
            #whichever molecule is bigger will have more eigenvalues returned
            difference = len(e_mol_1)-len(e_mol_2) #the difference in number of molecules between the two
            e_mol_2 = np.pad(e_mol_2, (0,difference), mode='constant', constant_values=0)
            #np.pad pads the smaller array with zeros, using however many atoms it is less by. ie if self is 6 atoms and argument is 4, np.pad will 
            #add 2 zeros to the argument eigenvalues array

        elif len(e_mol_2) > len(e_mol_1): #see comments above, same logic used here, except this time self is the smaller array
            difference = len(e_mol_2)-len(e_mol_1)
            e_mol_1 = np.pad(e_mol_1, (0,difference), mode='constant', constant_values=0)
            
        distance = np.linalg.norm(e_mol_1-e_mol_2) #np.linalg.norm returns the euclidean norm of the vector difference of the two molecules' sorted eigenvalues
        return distance
    
    def sortedCM(self):
        '''This methods returns the matrix in sorted order; atoms are reordered from highest norms to lowest norms in both rows and columns'''
        original_coulomb = self.rawCM() #takes the original coulomb matrix of the molecule
        eu_norms=np.linalg.norm(original_coulomb, axis=1) #this calculates the norms of the each row, row is done by setting axis to 1, a list of the
        #norms are returned
        sort_norms = np.argsort(eu_norms)[::-1] #the argsort function reorders the array in ascending order, but crucially it returns the indices of the
        #ordered numbers from the original array, not the numbers themselves. [::-1] then reorders these indices from back to front, descending order
        sorted_rows = original_coulomb[sort_norms,:] #the rows of the original coulomb are sorted, using the argsort indices, this moves the row with 
        #the highest norm (whose index would've been first in the argsort array) to row index 0
        sorted_matrix = sorted_rows[:,sort_norms] #as this is a symmetrical matrix, we need to apply the same sorting to the columns, in order to keep
        #the diagonal and off-diagonal interaction energies in their correct [i][j] positions

        return sorted_matrix

    def sortedCM_distance(self, mole_2):
        '''This methods returns the distance between two molecules using the frobenius norm of the two sorted coulomb matrices.'''
        s_mol_1 = self.sortedCM()
        s_mol_2 = mole_2.sortedCM()
        # difference = abs(self.natoms - mole_2.natoms)

        if self.natoms>mole_2.natoms: #this compares the number of atoms of both molecules, if the self molecule is bigger than the argument molecule
            difference=self.natoms-mole_2.natoms
            s_mol_2 = np.pad(s_mol_2, ((0,difference),(0,difference)),mode='constant', constant_values=0)
            #here, we are padding the array with zeros. the difference in number of atoms will be used to add that same number of rows and colomns
            #0 rows are added before row index [0], 0 columns are added before column index[0]. 

        elif self.natoms<mole_2.natoms: #uses the same logic as code above, except this time the self molecule is smaller, so it's matrix gets padded
            difference=mole_2.natoms-self.natoms
            s_mol_1 = np.pad(s_mol_1, ((0,difference),(0,difference)),mode='constant', constant_values=0)
        
        distance = np.linalg.norm(s_mol_1-s_mol_2, 'fro')
        return distance

