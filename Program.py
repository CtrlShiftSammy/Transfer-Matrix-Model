import cmath
import numpy as np
import matplotlib.pyplot as plt

# Define a function to create a 2x2 matrix for each layer of the material
def Mk_matrix(row_data, n1, theta1, lambda0):
    # Extract real and imaginary parts and thickness of the layer
    real_part, imag_part, dk = row_data
    # Calculate the complex permittivity of the layer
    eps = complex(real_part, imag_part)
    # Calculate qk, which is a term used in the matrix calculation
    qk = cmath.sqrt(eps - (n1 * cmath.sin(2 * cmath.pi * theta1 / 360)) ** 2) / eps
    # Calculate bk, which is another term used in the matrix calculation
    bk = 2 * cmath.pi * dk * cmath.sqrt(eps - (n1 * cmath.sin(2 * cmath.pi * theta1 / 360)) ** 2) / lambda0
    # Create a 2x2 matrix for this layer and fill in its values
    M_array = np.zeros((2,2),dtype=np.complex_)
    complex_i = 1j
    M_array[0,0] = cmath.cos(bk)
    M_array[0,1] = -1 * complex_i * cmath.sin(bk) / qk
    M_array[1,0] = -1 * complex_i * cmath.sin(bk) * qk
    M_array[1,1] = cmath.cos(bk)
    # Return the matrix
    return(M_array)

# Define a function to calculate qk for a given layer
def qk_val(row_data, n1, theta1, lambda0):
    # Extract real and imaginary parts of the layer and calculate its complex permittivity
    real_part, imag_part, dk = row_data
    eps = complex(real_part, imag_part)
    # Calculate qk for this layer
    qk = cmath.sqrt(eps - (n1 * cmath.sin(2 * cmath.pi * theta1 / 360)) ** 2) / eps
    # Return qk
    return qk

# Define function to calculate M(k) matrix for a given angle of incidence
def M_matrix(theta1):
    # Open input file
    file = open("Input.txt", "r")

    # Read in parameters from first line of input file
    params = file.readline().split()
    n1 = float(params[0])
    lambda0 = float(params[1])

    # Read in dimensions of data matrix from second line of input file
    dimensions = file.readline().split()
    N = int(dimensions[0]) # Number of layers
    num_cols = int(dimensions[1]) # Number of columns in data matrix

    # Read in data matrix from input file
    data_matrix = []
    for i in range(N):
        row = file.readline().split()
        row_data = [float(x) for x in row] # Convert each value to a float
        data_matrix.append(row_data)
    
    # Create a copy of the data matrix to pass to qk_val function
    data_matrix_copy = data_matrix.copy()
    
    # Close input file
    file.close()

    # Calculate M(k) matrices for each layer and store in a list
    M_array = []
    for row_index in range(1, N-1):
        row = data_matrix[row_index]
        Mk = Mk_matrix(row, n1, theta1, lambda0)
        M_array.append(Mk)

    # Calculate total M(k) matrix by multiplying all M(k) matrices together
    M = M_array[0]
    for data_matrix in M_array[1:]:
        M = np.matmul(M, data_matrix)

    # Calculate q1 and qn values using the first and last rows of the data matrix
    q1 = qk_val(data_matrix_copy[0], n1, theta1, lambda0)
    qn = qk_val(data_matrix_copy[N-1], n1, theta1, lambda0)

    # Return the total M(k) matrix and the q1 and qn values
    return M, q1, qn

# Define function to calculate reflection coefficient for a given incidence angle
def reflection_coeff(theta1):
    # Calculate M matrix, q1, and qn values for given incidence angle
    M, q1, qn = M_matrix(theta1)
    
    # Calculate reflection coefficient using formula
    rp = ((M[0,0] + M[0,1] * qn) * q1 - (M[1,0] + M[1,1] * qn)) / ((M[0,0] + M[0,1] * qn) * q1 + (M[1,0] + M[1,1] * qn))
    return(rp) 

# Define function to calculate reflectance for a given incidence angle
def reflectance(theta1):
    # Calculate reflection coefficient for given incidence angle
    r = np.abs(reflection_coeff(theta1)) ** 2
    
    # Print reflectance value to console
    print(r)
    
    return(r)

# Generate a list of incidence angles to calculate reflectance for
theta_values = range(0, 90, 1)

# Calculate reflectance for each incidence angle in the list
results = [reflectance(theta) for theta in theta_values]

# Plot reflectivity curve using matplotlib
plt.plot(theta_values, results)
plt.xlabel('Incidence angle (deg)')
plt.ylabel('Reflectivity')
plt.title('Reflectivity curve')
plt.show()
