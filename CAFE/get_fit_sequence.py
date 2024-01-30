#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import ipdb

def get_fit_sequence(image, snr_ind_seq=None, sorting_seq='snr', neighbor_dist=1.5, verbose=True, **kwargs):
        
        if verbose == True: print('Generating fitting sequence. Mode:',sorting_seq)

        image_copy = image.copy()
        # If the image is not masked then 
        if not isinstance(image_copy, np.ma.MaskedArray):
            image_copy = ma.masked_array(image_copy)
            image_copy.mask = False
            image_copy.fill_value = np.nan
            image_copy[image <= 0.] = ma.masked

        snr_image = image_copy.copy()
                
        # Store the 2D indices of the (masked) SNR image ranked from max to min SNR
        if snr_ind_seq == None: snr_ind_seq = np.unravel_index(np.flip(snr_image.argsort(axis=None,endwith=False)), snr_image.shape)
        if snr_image.mask[snr_ind_seq[0][0], snr_ind_seq[1][0]] == True: raise ValueError('The first element of the sequence is masked, which is a bad thing.')
        #snr_nunmasked = snr_image.count(axis=None)
        
        # Initialize dictionary containing a matrix that will store, in each pixel, the indices of the spaxel
        # to be used for the parameter initialization, as well as a tracking image used to know what spaxels
        # have been already "fitted" in previous steps
        param_ind_seq = {'parini_spx_ind': np.full((2,)+snr_image.shape, -99), 'track': np.full(snr_image.shape, False)}
        
        
        if sorting_seq == 'snr':
                
            x, y = np.meshgrid(np.arange(snr_image.shape[1]), np.arange(snr_image.shape[0]))
            
            # For each spaxel
            for snr_ind in zip(snr_ind_seq[0], snr_ind_seq[1]): # (y,x)
                
                if snr_image.mask[snr_ind] == False:
                    # First spaxel
                    if snr_ind == (snr_ind_seq[0][0], snr_ind_seq[1][0]):
                        ind_seq = (np.array([snr_ind[0]]), np.array([snr_ind[1]]))
                        param_ind_seq['parini_spx_ind'][:,snr_ind[0],snr_ind[1]] = snr_ind
                        param_ind_seq['track'][snr_ind] = True
                    else:
                        # Mask with closest neighbors
                        neighbors = np.sqrt((x-snr_ind[1])**2 + (y-snr_ind[0])**2) <= neighbor_dist
                        # Mask with closest neighbors that have been already fitted
                        fitted_neighbor_inds = np.logical_and(param_ind_seq['track'], neighbors)
                        # If there is any
                        if fitted_neighbor_inds.any():
                            # Chose the one with the highest SNR
                            snr_image[~fitted_neighbor_inds] = ma.masked
                            max_snr_fitted_neighbor_ind = np.where(snr_image == ma.max(snr_image))
                            snr_image.mask = image_copy.mask #if isinstance(image, np.ma.MaskedArray) else False 
                        else:
                            # Chose the first (highest SNR) spaxel in the image
                            max_snr_fitted_neighbor_ind = (np.array([snr_ind_seq[0][0]]), np.array([snr_ind_seq[1][0]]))
                            ## Assign its own index
                            #max_snr_fitted_neighbor_ind = (np.array([snr_ind[0]]), np.array([snr_ind[1]]))
                        # Assign the indices to the sequence
                        ind_seq = np.concatenate((ind_seq, (np.array([snr_ind[0]]),np.array([snr_ind[1]]))), axis=1)
                        param_ind_seq['parini_spx_ind'][:,snr_ind[0],snr_ind[1]] = np.concatenate(max_snr_fitted_neighbor_ind)
                        param_ind_seq['track'][snr_ind] = True
        
            # Returns the indices of the SNR sorted image, and the initiaziation indices of each spaxel
            return (ind_seq[0],ind_seq[1]), param_ind_seq['parini_spx_ind'].astype(int) # (y,x)
        
        
        elif sorting_seq == 'spiral':
            
            N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0) # directions
            turn_right = {N: E, E: S, S: W, W: N} # old -> new direction
            
            dx, dy = N # initial direction
            #matrix = [[None] * snr_image.shape[1] for _ in range(snr_image.shape[0])]
            count = 0
            param_ind_seq['parini_spx_ind'][:,snr_ind_seq[0][count],snr_ind_seq[1][count]] = np.array([snr_ind_seq[0][count], snr_ind_seq[1][count]])
            param_ind_seq['track'][snr_ind_seq[0][count],snr_ind_seq[1][count]] = True
            while True:
                #matrix[y][x] = count # visit
                # try to turn right
                new_dx, new_dy = turn_right[dx,dy]
                new_x, new_y = snr_ind_seq[1][count] + new_dx, snr_ind_seq[0][count] + new_dy
                if (0 <= new_x < snr_image.shape[1] and 0 <= new_y < snr_image.shape[0] and param_ind_seq['track'][new_y,new_x] == False):
                    #matrix[new_y][new_x] is None): # can turn right
                    count += 1
                    snr_ind_seq[1][count], snr_ind_seq[0][count] = new_x, new_y
                    param_ind_seq['parini_spx_ind'][:,snr_ind_seq[0][count],snr_ind_seq[1][count]] = np.array([snr_ind_seq[0][count-1], snr_ind_seq[1][count-1]])
                    param_ind_seq['track'][snr_ind_seq[0][count],snr_ind_seq[1][count]] = True
                    dx, dy = new_dx, new_dy
                else: # try to move straight
                    new_x, new_y = snr_ind_seq[1][count] + dx, snr_ind_seq[0][count] + dy
                    if not (0 <= new_x < snr_image.shape[1] and 0 <= new_y < snr_image.shape[0]):
                        #return matrix # nowhere to go
                        if len(snr_ind_seq[0]) != snr_image.shape[0]*snr_image.shape[1]: raise ValueError('The spiral pattern does not have as many elements as pixels in the total image')
                        
                        # Now call the sequencer to trim masked spaxels and assign initial spaxels based on SNR
                        return get_fit_sequence(image, snr_ind_seq=snr_ind_seq, sorting_seq='snr', neighbor_dist=neighbor_dist, verbose=False, **kwargs)
                        #return (snr_ind_seq[0],snr_ind_seq[1]), param_ind_seq['parini_spx_ind'].astype(int) # (y,x)
                    else:
                        count += 1
                        snr_ind_seq[1][count], snr_ind_seq[0][count] = new_x, new_y
                        param_ind_seq['parini_spx_ind'][:,snr_ind_seq[0][count],snr_ind_seq[1][count]] = np.array([snr_ind_seq[0][count-1], snr_ind_seq[1][count-1]])
                        param_ind_seq['track'][snr_ind_seq[0][count],snr_ind_seq[1][count]] = True



