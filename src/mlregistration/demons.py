import numpy as np
import SimpleITK as sitk


def demons_registration(fixed_image_np, moving_image_np, iterations=400, sigma=5):
    """
    Applies Demons non-rigid registration to align the moving image to the fixed image.

    Parameters:
    - fixed_image_np (numpy.ndarray): The fixed image (reference) as a numpy array.
    - moving_image_np (numpy.ndarray): The moving image to be aligned as a numpy array.
    - iterations (int): The number of iterations for the Demons algorithm.
    - sigma (float): The standard deviation for the Gaussian smoothing of the displacement field.

    Returns:
    - numpy.ndarray: The transformed moving image as a numpy array.
    """

    # Convert numpy arrays to SimpleITK images
    fixed_image = sitk.GetImageFromArray(fixed_image_np)
    moving_image = sitk.GetImageFromArray(moving_image_np)

    # Initialize the Demons registration method
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)
    demons_filter.SetStandardDeviations(sigma)

    # Perform the registration
    displacement_field = demons_filter.Execute(fixed_image, moving_image)

    # Resample the moving image to the fixed image space using the displacement field
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(sitk.DisplacementFieldTransform(displacement_field))
    resampler.SetInterpolator(sitk.sitkLinear)

    # Get the registered moving image
    registered_image = resampler.Execute(moving_image)

    # Convert the result back to numpy array
    registered_image_np = sitk.GetArrayFromImage(registered_image)

    return registered_image_np
