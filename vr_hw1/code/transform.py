import numpy as np


def identity(N):
    """
    Parameters:
    - N (int): size of the identity matrix.
    Return:
    - np.array, identity matrix of size NxN.
    """
    return np.identity(N, dtype=np.float32)


def zeros(N):
    """
    Parameters:
    - N (int): size of the zero matrix.
    Return:
    - np.array, zero matrix of size NxN.
    """
    return np.zeros((N, N), dtype=np.float32)


def normalize(v):
    """
    Parameters:
    - v (np.array): vector to be normalized.
    Return:
    - np.array, normalized unit vector.
    """
    return v / np.linalg.norm(v)


def translate(offsets):
    """
    Compute the 4x4 translation matrix.
    Parameters:
    - offsets (list of length 3): translation offsets. offsets[0], offsets[1], offsets[2] are offset on the x, y, z axis respectively.
    Return:
    - np.array, 4x4 translation matrix.
    """
    t_mat = identity(4)
    t_mat[0:3, 3] = offsets
    return t_mat


# -------------------------------------------#
#            Begin Assignment 1              #
# -------------------------------------------#

"""
Useful NumPy operations:
- degrees to radians: np.radians(angle)
- vector dot product: a.dot(b) or np.dot(a, b)
- vector cross product: np.cross(a, b)
- matrix multiplication: np.matmul(a, b) or a @ b
"""

def rotate(angle, axis):
    """
    Compute the 4x4 rotation matrix.
    Parameters:
    - angle (float): rotation angle, in degrees.
    - axis (list of length 3): the axis to rotate about.
    Return:
    - np.array, 4x4 rotation matrix.
    """
    
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    axis = np.array(axis)
    axis = normalize(axis)
    x, y, z = axis
    
    # calculate the rotation matrix with the formula
    rotation_matrix = np.array([
        [cos_theta + x**2 * (1 - cos_theta), x*y*(1 - cos_theta) - z*sin_theta, x*z*(1 - cos_theta) + y*sin_theta, 0],
        [y*x*(1 - cos_theta) + z*sin_theta, cos_theta + y**2 * (1 - cos_theta), y*z*(1 - cos_theta) - x*sin_theta, 0],
        [z*x*(1 - cos_theta) - y*sin_theta, z*y*(1 - cos_theta) + x*sin_theta, cos_theta + z**2 * (1 - cos_theta), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix
    
def scale(factors):
    """
    Compute the 4x4 scaling matrix.
    Parameters:
    - factors (list of length 3): scaling factors. factors[0], factors[1], factors[2] are scaling factor for the x, y, z axis respectively.
    Return:
    - np.array, 4x4 scaling matrix.
    """
    fx, fy, fz = factors
    
    matrix = np.array([[fx,  0,  0,  0],
                  [ 0, fy,  0,  0],
                  [ 0,  0, fz,  0],
                  [ 0,  0,  0,  1]])
    return matrix


def modelTransform(offsets, angles, factors):
    """
    Compute the model transformation matrix by applying transformations in the following order:
    1. scaling
    2. rotation about x(1, 0, 0)
    3. rotation about y(0, 1, 0)
    4. rotation about z(0, 0, 1)
    5. translation
    Call translate, rotate and scale implemented above to get a final model transformation matrix.
    Parameters:
    - offsets (list of length 3): translation offsets. offsets[0], offsets[1], offsets[2] are offset on the x, y, z axis respectively.
    - angles (list of length 3): rotation angles. angles[0], angles[1], angles[2] are rotation angle around the x, y, z axis respectively.
    - factors (list of length 3): scaling factors. factors[0], factors[1], factors[2] are scaling factor for the x, y, z axis respectively.
    Return:
    - np.array, 4x4 model transformation matrix.
    """
    
    scale_matrix = scale(factors)
    rotate_x = rotate(angles[0], [1, 0, 0])
    rotate_y = rotate(angles[1], [0, 1, 0])
    rotate_z = rotate(angles[2], [0, 0, 1])
    translate_matrix = translate(offsets)
    # do matrix multiplication in the order of scaling, rotation, translation
    model_matrix = np.matmul(translate_matrix, np.matmul(rotate_z, np.matmul(rotate_y, np.matmul(rotate_x, scale_matrix))))
    return model_matrix


def viewTransform(ori, center, up):
    """
    Compute the view transformation matrix given camera position, look-at point, and up vector.
    We assume the camera looks at -z and up at y in the camera coordinate.
    Parameters:
    - ori (list of length 3): camera position.
    - center (list of length 3): camera look-at point.
    - up (list of length 3): camera up direction.
    Return:
    - np.array, 4x4 view transformation matrix.
    """
    np_ori=np.array(ori)
    np_center=np.array(center)
    np_up=np.array(up)
    
    z_axis=normalize(np_ori-np_center) # 算出来的 z 轴
    x_axis=normalize(np.cross(np_up,z_axis)) # x
    y_axis=np.cross(z_axis,x_axis) # y
    
    # use the formula to calculate the view matrix 
    view_matrix = np.array([
        [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis,np_ori)],
        [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis,np_ori)],
        [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis,np_ori)],
        [0, 0, 0, 1]
    ])
    return view_matrix


def perspectiveProjection(fovy, aspect, zNear, zFar):
    """
    Compute the perspective projection transformation matrix.
    Parameters:
    - fovy (float): field of view in y direction, in degrees.
    - aspect: aspect ratio of the image (width / height).
    - zNear: z value of the near clipping plane.
    - zFar: z value of the far clipping plane.
    Return:
    - np.array, 4x4 perspective projection transformation matrix.
    """
    fovy_rad = np.radians(fovy)
    
    f = 1 / np.tan(fovy_rad / 2)
    a = aspect
    
    # use formula to calculate the perspective matrix
    perspective_matrix = np.array([
        [f/a, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (zFar + zNear) / (zNear - zFar), 2 * zFar * zNear / (zNear - zFar)],
        [0, 0, -1, 0]
    ])
    return perspective_matrix


def orthogonalProjection(left, right, bottom, top, zNear, zFar):
    """
    Compute the orthogonal projection transformation matrix.
    Parameters:
    - left: x value of the left side of the near clipping plane.
    - right: x value of the right side of the near clipping plane.
    - bottom: y value of the bottom side of the near clipping plane.
    - top: y value of the top side of the near clipping plane.
    - zNear: z value of the near clipping plane.
    - zFar: z value of the far clipping plane.
    Return:
    - np.array, 4x4 orthogonal projection transformation matrix.
    """
    scale_x = 2.0 / (right - left)
    scale_y = 2.0 / (top - bottom)
    scale_z = -2.0 / (zFar - zNear)
    
    trans_x = -(right + left) / (right - left)
    trans_y = -(top + bottom) / (top - bottom)
    trans_z = -(zFar + zNear) / (zFar - zNear)
    
    ortho_matrix = np.array([
        [scale_x,       0,          0,         trans_x],
        [0,         scale_y,        0,         trans_y],
        [0,             0,      scale_z,       trans_z],
        [0,             0,          0,            1]
    ])
    
    return ortho_matrix
# -------------------------------------------#
#              End Assignment 1              #
# -------------------------------------------#
