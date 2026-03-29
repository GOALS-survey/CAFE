# cafe_cubefit_parallel.py

import sys
import os
import time
import zipfile

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from astropy.utils.data import download_file
from astropy.io import ascii, fits
from astropy.table import Table, QTable

import importlib as imp
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# CAFE imports (refactored package names)
from cafe.io import cafe_io
from cafe.fitter import specmod, cubemod
from cafe.lib import *
from cafe.params import CAFE_param_generator, CAFE_prof_generator
from cafe.get_fit_sequence import get_fit_sequence, plot_fit_sequence
from cafe.paths import get_package_data_path

# CRETA import
from creta.extractor import creta

# Add current directory to Python path (for local helper scripts)
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from STcube2CretaOutput import ST_nirspec2creta_output

cafeio = cafe_io()


def prepare_cafe_cube(
    input_fits,
    output_dir,
    sorting_seq="snr",
    min_valid_pixels=500,
    snr_slice=(240, 250),
    make_skip_plot=True,
):
    """
    Prepare a JWST NIRSpec ST cube for CAFE:
      - convert ST cube to CRETA output
      - compute fit sequence
      - build spaxel skip list and mask
      - initialize keys, fitted_hdu

    Returns a dict with everything needed for the fitting step.
    """

    # 1. Convert the ST cube to the CRETA output format
    creta_fits = ST_nirspec2creta_output(input_fits)
    print(f"CRETA fits file has been created: {creta_fits}")

    source_fd = os.path.dirname(creta_fits)
    source_fn = os.path.basename(creta_fits)

    hdul = fits.open(os.path.join(source_fd, source_fn))

    # 2. Build an "intensity" image for defining the fit sequence
    kmin, kmax = snr_slice
    int_image = np.nansum(hdul["FLUX_ST"].data[kmin:kmax, :, :], axis=0)

    # 3. Get the fit sequence
    ind_seq, ref_ind_seq = get_fit_sequence(int_image, sorting_seq=sorting_seq)

    # 4. Plot the fit sequence (optional)
    plot_fit_sequence(ind_seq, output_dir=output_dir)

    # 5. Initialize CAFE cube IO (this fills CAFE internal state)
    cafeio.read_cretacube(
        os.path.join(source_fd, source_fn),
        "Flux_st",
        "ERR_st",
    )

    # 6. Define keys and arrays
    dust_key = ["pah33", "ali34", "ali345"]
    line_key = ["pfund105", "pfund95", "pfund85", "brackett54"]

    keys = dust_key + line_key
    keys_unc = [k + "_unc" for k in keys]
    all_keys = keys + keys_unc

    fitted_hdu = np.zeros(hdul["FLUX_ST"].data.shape[1:], dtype=int)

    # 7. Build the spaxel skip list
    spx_skip_list = []

    x_range = range(np.min(ind_seq[1]), np.max(ind_seq[1]) + 1)
    y_range = range(np.min(ind_seq[0]), np.max(ind_seq[0]) + 1)

    for x in x_range:
        for y in y_range:
            arr = hdul["FLUX_ST"].data[:, y, x]
            # If the spectrum is all NaN or has too few good pixels, skip the spaxel
            if (np.isnan(np.unique(arr)[0])) or (
                len(arr[~np.isnan(arr)]) < min_valid_pixels
            ):
                spx_skip_list.append([x, y])

    # 8. Make and optionally plot the skip mask
    nx, ny = fitted_hdu.shape[1], fitted_hdu.shape[0]
    mask = np.zeros((nx, ny))
    for x, y in spx_skip_list:
        mask[x, y] = 1

    if make_skip_plot:
        plt.figure()
        plt.imshow(np.transpose(mask), origin="lower", cmap="gray")
        plt.title("Spaxels in spx_skip_list")
        cbar = plt.colorbar(label="Mask value")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["not skipped", "skipped"])
        plt.savefig(os.path.join(output_dir, "spx_skip_list.png"))

    return {
        "creta_fits": creta_fits,
        "source_fd": source_fd,
        "source_fn": source_fn,
        "hdul": hdul,
        "ind_seq": ind_seq,
        "ref_ind_seq": ref_ind_seq,
        "spx_skip_list": spx_skip_list,
        "dust_key": dust_key,
        "line_key": line_key,
        "all_keys": all_keys,
        "fitted_hdu": fitted_hdu,
    }


def fit_chunk_worker(
    worker_id,
    spaxels,  # list of (x, y)
    source_fd,
    source_fn,
    inppar_fn,
    optpar_fn,
    dust_key,
    line_key,
    all_keys,
    template_hdul_path,  # path to original CRETA cube to clone header/WCS
    output_dir,
    z=0.0,
    autosave_interval=40,
):
    # Each worker has its own specmod (no cafe_dir needed)
    s = specmod()

    # Open a template HDUList to copy headers from
    template_hdul = fits.open(template_hdul_path)
    ny, nx = template_hdul["FLUX_ST"].data.shape[1:]

    # Local maps only for this worker
    result_map = {k: np.full((ny, nx), np.nan) for k in all_keys}
    fitted_hdu = np.zeros((ny, nx), dtype=int)

    # make a worker directory to save the intermediate results if not already exists
    worker_dir = os.path.join(output_dir, "worker_fits_files")
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)

    worker_prefix = os.path.splitext(source_fn)[0] + f"_worker{worker_id}"
    worker_autosave = os.path.join(worker_dir, f"{worker_prefix}_autosave.fits")
    worker_final = os.path.join(worker_dir, f"{worker_prefix}_final.fits")

    start = time.time()

    for i, (x, y) in enumerate(spaxels):
        print(
            f"[worker {worker_id}] fitting {i+1}/{len(spaxels)} at (x={x}, y={y})"
        )

        # Read spectrum
        s.read_spec(
            source_fn,
            xy=(x, y),
            file_dir=source_fd,
            z=z,
            rwave_min=2.9,
            rwave_max=4.25,
        )
        s.input_param(inppar_fn, optpar_fn)

        try:
            s.fit_spec()
        except Exception as e:
            print(f"[worker {worker_id}] error at ({x},{y}): {e}")
            continue

        pah = s.pahtable
        line = s.linetable

        # Dust
        for dk in dust_key:
            idx = pah.index.str.lower().str.contains(dk)
            result_map[dk][y, x] = (
                pah.pah_strength_obs[idx].iloc[0] if idx.any() else 0.0
            )
            result_map[dk + "_unc"][y, x] = (
                pah.pah_strength_obs_unc[idx].iloc[0] if idx.any() else 0.0
            )

        # Lines
        for lk in line_key:
            idx = line.index.str.contains(lk)
            result_map[lk][y, x] = (
                line.line_strength_obs[idx].iloc[0] if idx.any() else 0.0
            )
            result_map[lk + "_unc"][y, x] = (
                line.line_strength_obs_unc[idx].iloc[0] if idx.any() else 0.0
            )

        fitted_hdu[y, x] = 1

        # Worker-local autosave
        if (
            (i == 0)
            or ((i + 1) % autosave_interval == 0)
            or ((i + 1) == len(spaxels))
        ):
            _write_worker_fits(
                template_hdul,
                result_map,
                fitted_hdu,
                worker_autosave,
            )
            print(
                f"[worker {worker_id}] autosaved to {worker_autosave} at i={i+1}"
            )

    # Final worker file (could just reuse the last autosave path)
    _write_worker_fits(template_hdul, result_map, fitted_hdu, worker_final)
    print(f"[worker {worker_id}] finished in {time.time() - start:.1f}s")

    template_hdul.close()
    return worker_final


def _write_worker_fits(template_hdul, result_map, fitted_hdu, out_path):
    # Clone template_hdul but drop extra extensions; keep primary + FLUX_ST, etc.
    hdul_out = fits.HDUList()
    for h in template_hdul:
        hdul_out.append(h.copy())

    existing_extnames = [h.name for h in hdul_out]

    # Add/update science extensions
    for k, arr in result_map.items():
        if k in existing_extnames:
            hdul_out[k].data[...] = arr
        else:
            new_hdu = fits.ImageHDU(data=arr)
            new_hdu.header = hdul_out[1].header.copy()
            new_hdu.header["EXTNAME"] = k
            # strip 3D WCS
            for k3d in [
                "PC3_1",
                "PC2_3",
                "CTYPE3",
                "PC1_3",
                "CRVAL3",
                " CUNIT3",
                "CDELT3",
                "PC3_3",
                "PC3_2",
                "CRPIX3",
            ]:
                new_hdu.header.pop(k3d, None)
            new_hdu.header["WCSAXES"] = 2
            hdul_out.append(new_hdu)

    # FIT_FLAG
    if "FIT_FLAG" in existing_extnames:
        hdul_out["FIT_FLAG"].data = fitted_hdu
    else:
        hdu_flag = fits.ImageHDU(data=fitted_hdu)
        hdu_flag.header = hdul_out[1].header.copy()
        hdu_flag.header["EXTNAME"] = "FIT_FLAG"
        hdul_out.append(hdu_flag)

    hdul_out.writeto(out_path, overwrite=True)


def merge_worker_results(
    template_hdul_path,
    worker_final_paths,
    all_keys,
    output_dir,
    final_output_fn,
):
    template_hdul = fits.open(template_hdul_path)
    ny, nx = template_hdul["FLUX_ST"].data.shape[1:]

    # Global maps
    result_map = {k: np.full((ny, nx), np.nan) for k in all_keys}
    fitted_hdu = np.zeros((ny, nx), dtype=int)

    # Read each worker file and OR/overwrite
    for path in worker_final_paths:
        whdul = fits.open(path)
        # Combine FIT_FLAG
        if "FIT_FLAG" in whdul:
            fitted_hdu = np.maximum(fitted_hdu, whdul["FIT_FLAG"].data)

        for k in all_keys:
            if k in whdul:
                data = whdul[k].data
                mask = ~np.isnan(data)
                result_map[k][mask] = data[mask]
        whdul.close()

    # Write final FITS
    final_path = os.path.join(output_dir, final_output_fn)
    hdul_out = fits.HDUList([h.copy() for h in template_hdul])

    existing_extnames = [h.name for h in hdul_out]

    for k, arr in result_map.items():
        if k in existing_extnames:
            hdul_out[k].data[...] = arr
        else:
            new_hdu = fits.ImageHDU(data=arr)
            new_hdu.header = hdul_out[1].header.copy()
            new_hdu.header["EXTNAME"] = k
            for k3d in [
                "PC3_1",
                "PC2_3",
                "CTYPE3",
                "PC1_3",
                "CRVAL3",
                " CUNIT3",
                "CDELT3",
                "PC3_3",
                "PC3_2",
                "CRPIX3",
            ]:
                new_hdu.header.pop(k3d, None)
            new_hdu.header["WCSAXES"] = 2
            hdul_out.append(new_hdu)

    # FIT_FLAG
    if "FIT_FLAG" in existing_extnames:
        hdul_out["FIT_FLAG"].data = fitted_hdu
    else:
        hdu_flag = fits.ImageHDU(data=fitted_hdu)
        hdu_flag.header = hdul_out[1].header.copy()
        hdu_flag.header["EXTNAME"] = "FIT_FLAG"
        hdul_out.append(hdu_flag)

    hdul_out.writeto(final_path, overwrite=True)
    print(f"Final merged cube written to {final_path}")

    template_hdul.close()
    return final_path


def prepare_spaxels_for_workers(
    ind_seq,
    spx_skip_list,
    fitted_hdu_xy_list,
    num_for_test=None,
    n_workers=None,
):
    """
    Filter target spaxels and split them into worker chunks.

    Returns
    -------
    chunks : list of lists of (x, y)
    target_spaxels : list of (x, y)
    """

    # Convert skip lists to sets for fast membership lookup
    skip_set = set(tuple(p) for p in spx_skip_list)
    fitted_set = set(tuple(p) for p in fitted_hdu_xy_list)

    # Unpack sequence safely
    yy, xx = ind_seq  # explicit names!
    target_spaxels = [
        (int(x), int(y))
        for x, y in zip(xx, yy)
        if (x, y) not in skip_set and (x, y) not in fitted_set
    ]

    # Apply testing limit
    if num_for_test is not None:
        target_spaxels = target_spaxels[:num_for_test]

    # Even splitting across workers
    chunks = [target_spaxels[i::n_workers] for i in range(n_workers)]

    return chunks, target_spaxels


def run_cafe_cube_fit_mapreduce(
    input_fits,
    inppar_fn,
    optpar_fn,
    output_dir,
    z=0.0,
    n_workers=4,
    autosave_interval=40,
    num_for_test=None,
    min_valid_pixels=500,
):
    start = time.time()

    # prepare_cafe_cube returns the "prep" dict you already defined
    prep = prepare_cafe_cube(
        input_fits=input_fits,
        output_dir=output_dir,
        sorting_seq="snr",
        min_valid_pixels=min_valid_pixels,
    )

    source_fd = prep["source_fd"]
    source_fn = prep["source_fn"]
    ind_seq = prep["ind_seq"]
    spx_skip = prep["spx_skip_list"]
    all_keys = prep["all_keys"]
    dust_key = prep["dust_key"]
    line_key = prep["line_key"]
    fitted_hdu = prep["fitted_hdu"]
    fitted_xy = [[x, y] for y, x in np.argwhere(fitted_hdu == 1)]

    chunks, target_spaxels = prepare_spaxels_for_workers(
        ind_seq,
        spx_skip,
        fitted_xy,
        num_for_test=num_for_test,
        n_workers=n_workers,
    )

    print(f"Total spaxels to fit: {len(target_spaxels)}")
    print(f"Splitting into {n_workers} workers")

    template_hdul_path = os.path.join(source_fd, source_fn)
    worker_results = []

    if n_workers == 1:
        # serial path for debugging
        worker_final = fit_chunk_worker(
            worker_id=0,
            spaxels=target_spaxels,
            source_fd=source_fd,
            source_fn=source_fn,
            inppar_fn=inppar_fn,
            optpar_fn=optpar_fn,
            dust_key=dust_key,
            line_key=line_key,
            all_keys=all_keys,
            template_hdul_path=template_hdul_path,
            output_dir=output_dir,
            z=z,
            autosave_interval=autosave_interval,
        )
        worker_results = [worker_final]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = [
                exe.submit(
                    fit_chunk_worker,
                    wid,
                    chunks[wid],
                    source_fd,
                    source_fn,
                    inppar_fn,
                    optpar_fn,
                    dust_key,
                    line_key,
                    all_keys,
                    template_hdul_path,
                    output_dir,
                    z,
                    autosave_interval,
                )
                for wid in range(n_workers)
                if len(chunks[wid]) > 0
            ]

            for fut in as_completed(futures):
                worker_results.append(fut.result())

    final_output_fn = os.path.splitext(source_fn)[0] + "_pah33_cubefit.fits"
    final_path = merge_worker_results(
        template_hdul_path,
        worker_results,
        all_keys,
        output_dir,
        final_output_fn,
    )

    # Print the total run time
    print(f"Total run time: {time.time() - start:.1f}s")

    return final_path
