import vtk
import numpy as np


def thin_plate_spline_registration(fixed_keypoints, moving_keypoints, moving_image_np):
    """
    Applies Thin Plate Spline (TPS) transform to align the moving image based on keypoints.

    Parameters:
    - fixed_keypoints (numpy.ndarray): The keypoints from the fixed image as a numpy array of shape (N, 2).
    - moving_keypoints (numpy.ndarray): The keypoints from the moving image as a numpy array of shape (N, 2).
    - moving_image_np (numpy.ndarray): The moving image to be aligned as a numpy array.

    Returns:
    - numpy.ndarray: The transformed moving image as a numpy array.
    """

    def numpy_to_vtk_array(numpy_array):
        """Convert a NumPy array to a VTK array."""
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(numpy_array.size)
        for i in range(numpy_array.size):
            vtk_array.SetTuple1(i, numpy_array.flat[i])
        return vtk_array

    def vtk_to_numpy_array(vtk_array, shape):
        """Convert a VTK array to a NumPy array."""
        numpy_array = np.zeros(vtk_array.GetNumberOfTuples(), dtype=np.float32)
        for i in range(vtk_array.GetNumberOfTuples()):
            numpy_array[i] = vtk_array.GetTuple1(i)
        return numpy_array.reshape(shape)

    # Convert numpy keypoints to vtkPoints
    def numpy_to_vtk_points(keypoints):
        vtk_points = vtk.vtkPoints()
        for kp in keypoints:
            vtk_points.InsertNextPoint(kp[0], kp[1], 0)
        return vtk_points

    fixed_points = numpy_to_vtk_points(fixed_keypoints)
    moving_points = numpy_to_vtk_points(moving_keypoints)

    # Create a Thin Plate Spline Transform
    tps_transform = vtk.vtkThinPlateSplineTransform()
    tps_transform.SetSourceLandmarks(moving_points)
    tps_transform.SetTargetLandmarks(fixed_points)

    # Create a VTK image
    moving_image_vtk = vtk.vtkImageData()
    moving_image_vtk.SetDimensions(
        moving_image_np.shape[1], moving_image_np.shape[0], 1
    )
    moving_image_vtk.SetSpacing(1, 1, 1)
    moving_image_vtk.SetOrigin(0, 0, 0)

    # Fill the image data
    vtk_array = numpy_to_vtk_array(moving_image_np.ravel())
    moving_image_vtk.GetPointData().SetScalars(vtk_array)

    # Apply the TPS transformation
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(moving_image_vtk)
    reslicer.SetResliceTransform(tps_transform)
    reslicer.SetOutputSpacing(1, 1, 1)
    reslicer.SetOutputOrigin(0, 0, 0)
    reslicer.SetInterpolationModeToLinear()

    # Update and get the transformed image
    reslicer.Update()
    transformed_image_vtk = reslicer.GetOutput()
    transformed_image_np = vtk_to_numpy_array(
        transformed_image_vtk.GetPointData().GetScalars(), moving_image_np.shape
    )

    return transformed_image_np


def create_sample_image(size=(100, 100)):
    """Create a sample image with a distinct pattern."""
    image = np.zeros(size, dtype=np.float32)
    image[30:70, 30:70] = 1  # A white square
    return image


def create_keypoints():
    """Create sample keypoints for fixed and moving images."""
    fixed_keypoints = np.array(
        [[30, 30], [30, 70], [70, 30], [70, 70]], dtype=np.float32
    )
    # Introduce a realistic transformation: translation + rotation
    transformation_matrix = np.array([[1.0, 0.1], [-0.1, 1.0]], dtype=np.float32)
    moving_keypoints = np.dot(
        fixed_keypoints - np.mean(fixed_keypoints, axis=0), transformation_matrix
    ) + np.mean(fixed_keypoints, axis=0)
    return fixed_keypoints, moving_keypoints


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create sample data
    fixed_image_np = create_sample_image()
    moving_image_np = create_sample_image()

    # Create keypoints
    fixed_keypoints, moving_keypoints = create_keypoints()

    # Apply TPS registration
    transformed_image_np = thin_plate_spline_registration(
        fixed_keypoints, moving_keypoints, moving_image_np
    )

    # Visualize results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Fixed Image")
    plt.imshow(fixed_image_np, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Moving Image")
    plt.imshow(moving_image_np, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(transformed_image_np, cmap="gray")

    plt.show()
