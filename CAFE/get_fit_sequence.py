#!/usr/bin/env python3

"""
This module contains the function get_fit_sequence, which generates a sequence of pixel
indices for fitting an image
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os


def get_fit_sequence(
    image,
    snr_ind_seq=None,
    sorting_seq="snr",
    neighbor_dist=1.5,
    verbose=True,
    **kwargs,
):
    """
    Parameters
    ----------
    image : numpy.ndarray
        2D array of the image to be fitted.
    snr_ind_seq : numpy.ndarray, optional
        2D array of pre-defined indices for the fitting sequence. If None, indices
        will be generated based on the sorting_seq parameter.
    sorting_seq : {'snr', 'spiral'}, default='snr'
        Method to determine fitting sequence:
        - 'snr': Sort pixels by signal-to-noise ratio
        - 'spiral': Process pixels in a spiral pattern from center
    neighbor_dist : float, default=1.5
        Maximum distance to consider neighboring pixels for parameter initialization.
    verbose : bool, default=True
        If True, print information about the sorting sequence mode.

    Returns
    -------
    tuple
        ((y_indices, x_indices), param_init_indices)
        - y_indices, x_indices: Arrays of pixel coordinates in fitting order
        - param_init_indices: 2D array of indices used for parameter initialization

    Raises
    ------
    ValueError
        If the first element in the sequence is masked or if the spiral pattern
        doesn't cover all image pixels.
    """
    print("Generating fitting sequence. Mode:", sorting_seq)

    image_copy = image.copy()

    # If the image has not masked values, then create a masked array
    if not isinstance(image_copy, np.ma.MaskedArray):
        image_copy = ma.masked_array(image_copy)
        image_copy.mask = False
        image_copy.fill_value = np.nan
        image_copy[(image <= 0.0) | np.isnan(image)] = ma.masked

    snr_image = image_copy.copy()

    # Store the 2D indices of the (masked) SNR image ranked from max to min SNR
    # if snr_ind_seq is None:
    snr_ind_seq = np.unravel_index(
        np.flip(snr_image.argsort(axis=None, endwith=False)),
        snr_image.shape,
    )

    if snr_image.mask[snr_ind_seq[0][0], snr_ind_seq[1][0]] is True:
        raise ValueError(
            "Invalid sequence: First element is masked. The sequence must begin with an unmasked value."
        )

    # Initialize a dictionary containing a matrix that will store, in each pixel, the indices of the spaxel
    # to be used for the parameter initialization, as well as a tracking image used to know what spaxels
    # have been already "fitted" in previous steps
    param_ind_seq = {
        "parini_spx_ind": np.full(
            (2,) + snr_image.shape, -99
        ),  # This is a 3D array with the first dimension being the spaxel index, and the second and third dimensions being the y and x coordinates
        "track": np.full(snr_image.shape, False),
    }

    if sorting_seq not in ["snr", "spiral"]:
        raise ValueError(f"Sorting sequence {sorting_seq} not supported")

    if sorting_seq == "snr":
        x, y = np.meshgrid(
            np.arange(snr_image.shape[1]), np.arange(snr_image.shape[0])
        )

        total_pixels = np.sum(~snr_image.mask)
        processed_pixels = 0
        # For each spaxel
        for snr_ind in zip(snr_ind_seq[0], snr_ind_seq[1]):  # (y,x)
            if snr_image.mask[snr_ind] == False:
                # Print the progress
                processed_pixels += 1
                percentage = (processed_pixels / total_pixels) * 100
                print(
                    f"\rProcessing get cube fitting sequence {processed_pixels}/{total_pixels} ({percentage:.1f}%)",
                    end="",
                    flush=True,
                )

                if snr_ind == (
                    snr_ind_seq[0][0],
                    snr_ind_seq[1][0],
                ):  # For the first spaxel
                    ind_seq = tuple(np.array([coord]) for coord in snr_ind)
                    param_ind_seq["parini_spx_ind"][
                        :, snr_ind[0], snr_ind[1]
                    ] = snr_ind
                    param_ind_seq["track"][snr_ind] = True
                else:  # For the rest of the spaxels
                    # Mask with closest neighbors
                    neighbors = (
                        np.sqrt((x - snr_ind[1]) ** 2 + (y - snr_ind[0]) ** 2)
                        <= neighbor_dist
                    )
                    # Mask with closest neighbors that have been already fitted
                    fitted_neighbor_inds = np.logical_and(
                        param_ind_seq["track"], neighbors
                    )
                    # If there is any neighbor that has not been already fitted
                    if fitted_neighbor_inds.any():
                        # Chose the one with the highest SNR
                        snr_image[~fitted_neighbor_inds] = ma.masked
                        t = np.where(snr_image == ma.max(snr_image))
                        # This is to handle the scenario where there are multiple pixels with the same maximum SNR
                        max_snr_fitted_neighbor_ind = (
                            np.array([t[0][0]]),
                            np.array([t[1][0]]),
                        )
                        snr_image.mask = (
                            image_copy.mask
                        )  # if isinstance(image, np.ma.MaskedArray) else False
                    else:
                        # Chose the first (highest SNR) spaxel in the image
                        max_snr_fitted_neighbor_ind = (
                            np.array([snr_ind_seq[0][0]]),
                            np.array([snr_ind_seq[1][0]]),
                        )
                        ## Assign its own index
                        # max_snr_fitted_neighbor_ind = (np.array([snr_ind[0]]), np.array([snr_ind[1]]))

                    # Assign the indices to the sequence
                    ind_seq = np.concatenate(
                        (
                            ind_seq,
                            (np.array([snr_ind[0]]), np.array([snr_ind[1]])),
                        ),
                        axis=1,
                    )
                    param_ind_seq["parini_spx_ind"][
                        :, snr_ind[0], snr_ind[1]
                    ] = np.concatenate(max_snr_fitted_neighbor_ind)
                    param_ind_seq["track"][snr_ind] = True
            else:
                continue

        # Returns the indices of the SNR sorted image, and the initiaziation indices of each spaxel
        output_ind_seq = (ind_seq[0], ind_seq[1])  # (y,x)
        output_param_ind_seq = param_ind_seq["parini_spx_ind"].astype(
            int
        )  # 3D array

        return output_ind_seq, output_param_ind_seq

    elif sorting_seq == "spiral":
        N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)  # directions
        turn_right = {N: E, E: S, S: W, W: N}  # old -> new direction

        dx, dy = N  # initial direction
        count = 0
        param_ind_seq["parini_spx_ind"][
            :, snr_ind_seq[0][count], snr_ind_seq[1][count]
        ] = np.array([snr_ind_seq[0][count], snr_ind_seq[1][count]])
        param_ind_seq["track"][
            snr_ind_seq[0][count], snr_ind_seq[1][count]
        ] = True

        while True:
            new_dx, new_dy = turn_right[dx, dy]
            new_x, new_y = (
                snr_ind_seq[1][count] + new_dx,
                snr_ind_seq[0][count] + new_dy,
            )
            if (
                0 <= new_x < snr_image.shape[1]
                and 0 <= new_y < snr_image.shape[0]
                and param_ind_seq["track"][new_y, new_x] == False
            ):
                count += 1
                snr_ind_seq[1][count], snr_ind_seq[0][count] = new_x, new_y
                param_ind_seq["parini_spx_ind"][
                    :, snr_ind_seq[0][count], snr_ind_seq[1][count]
                ] = np.array(
                    [snr_ind_seq[0][count - 1], snr_ind_seq[1][count - 1]]
                )
                param_ind_seq["track"][
                    snr_ind_seq[0][count], snr_ind_seq[1][count]
                ] = True
                dx, dy = new_dx, new_dy
            else:
                new_x, new_y = (
                    snr_ind_seq[1][count] + dx,
                    snr_ind_seq[0][count] + dy,
                )
                if not (
                    0 <= new_x < snr_image.shape[1]
                    and 0 <= new_y < snr_image.shape[0]
                ):
                    if (
                        len(snr_ind_seq[0])
                        != snr_image.shape[0] * snr_image.shape[1]
                    ):
                        raise ValueError(
                            "The spiral pattern does not have as many elements as pixels in the total image"
                        )

                    # Return from inside the function
                    return get_fit_sequence(
                        image,
                        snr_ind_seq=snr_ind_seq,
                        sorting_seq="snr",
                        neighbor_dist=neighbor_dist,
                        verbose=False,
                        **kwargs,
                    )
                else:
                    count += 1
                    snr_ind_seq[1][count], snr_ind_seq[0][count] = new_x, new_y
                    param_ind_seq["parini_spx_ind"][
                        :, snr_ind_seq[0][count], snr_ind_seq[1][count]
                    ] = np.array(
                        [snr_ind_seq[0][count - 1], snr_ind_seq[1][count - 1]]
                    )
                    param_ind_seq["track"][
                        snr_ind_seq[0][count], snr_ind_seq[1][count]
                    ] = True


def plot_fit_sequence(ind_seq, output_dir=None):
    """
    Plot the fit sequence with colors representing the order of fitting.

    Parameters
    ----------
    ind_seq : tuple
        Tuple of (y_indices, x_indices) arrays representing the fitting sequence
    """
    # Create empty array with shape of the image
    sequence_map = np.full(
        (np.max(ind_seq[0]) + 1, np.max(ind_seq[1]) + 1), np.nan
    )

    # Fill array with sequence index
    for i, (y, x) in enumerate(zip(ind_seq[0], ind_seq[1])):
        sequence_map[y, x] = i

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(sequence_map, cmap="viridis_r", origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Fitting sequence index", rotation=-90, va="bottom")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, "fit_sequence.png"))
