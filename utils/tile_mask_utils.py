import cv2
import numpy as np
import torch

def generate_pixel_mask(image_tensor, tile_size, sparse_fn, reduce_fn = 'normal', num_to_keep = 10000, **args):
    with torch.no_grad():
        if sparse_fn == 'uniform':
            mask, intensity_map = generate_tile_center_mask(image_tensor.shape, tile_size, **args)
        elif sparse_fn == 'fast':
            mask, intensity_map = generate_tile_fast_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'dog':
            mask, intensity_map = generate_tile_dog_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'harris':
            mask, intensity_map = generate_tile_harris_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'goodFeatures':
            mask, intensity_map = generate_tile_eig_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'orb':
            mask, intensity_map = generate_tile_orb_max_response_mask(image_tensor, tile_size, **args)
        elif sparse_fn == 'random':
            mask, intensity_map = generate_tile_random_mask(image_tensor, tile_size, **args)

        if reduce_fn == 'sort':
            mask = filter_masks_by_sort(mask, intensity_map, num_to_keep=num_to_keep)
            
    return mask

def filter_masks_by_sort(mask, intensity_map, num_to_keep=10000):
    """
    Filter the masks by sorting them based on their intensity values and 
    keeping only the top ratio of pixels.
    
    Args:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W).
        intensity_map (torch.Tensor): Float tensor of shape (H, W) with Harris corner intensities.
        ratio (float): The ratio of top intensity pixels to keep (0.0 to 1.0).
    
    Returns:
        filtered_mask (torch.Tensor): Filtered boolean mask tensor of shape (H, W).
    """
    masked_intensities = intensity_map[mask]
    
    sorted_indices = torch.argsort(masked_intensities, descending=True)
    
    # num_to_keep = int(len(sorted_indices) * ratio)
    if num_to_keep > int(len(sorted_indices)):
        return mask
    
    filtered_mask = torch.zeros_like(mask)
    
    if num_to_keep > 0:
        keep_indices = sorted_indices[:num_to_keep]
        
        row_indices, col_indices = torch.where(mask)
        
        mask_indices = (row_indices[keep_indices], col_indices[keep_indices])
        
        filtered_mask[mask_indices] = True
    
    return filtered_mask

def generate_tile_random_mask(image_tensor, tile_size, device='cuda'):
    """
    Generate random mask with one random pixel selected per tile.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor with shape (H, W, C)
        tile_size (Tuple[int, int]): Size of each tile (height, width)
        device (str): Device to place tensors on
        
    Returns:
        mask (torch.Tensor): Boolean mask tensor with shape (H, W)
        intensity_map (torch.Tensor): Float intensity map with shape (H, W)
    """
    if image_tensor.shape[0] == 3:  # C,H,W -> H,W,C
        C, H, W = image_tensor.shape
    else:
        H, W, C = image_tensor.shape
    th, tw = tile_size
    
    # Calculate number of tiles (round down)
    h_tiles = H // th
    w_tiles = W // tw
    
    # Generate random coordinates within each tile
    tile_y = np.random.randint(0, th, size=(h_tiles, w_tiles))
    tile_x = np.random.randint(0, tw, size=(h_tiles, w_tiles))
    
    # Create global coordinates by adding tile offsets
    y_coords = (np.arange(h_tiles)[:, None] * th + tile_y).flatten()
    x_coords = (np.arange(w_tiles)[None, :] * tw + tile_x).flatten()
    
    # Initialize output arrays
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Set selected pixels
    mask_np[y_coords, x_coords] = True
    intensity_np[y_coords, x_coords] = np.random.random(size=len(y_coords))
    
    # Convert to PyTorch tensors
    mask = torch.from_numpy(mask_np).to(device)
    intensity_map = torch.from_numpy(intensity_np).to(device)
    
    return mask, intensity_map

def generate_tile_dog_max_response_mask(image_tensor, tile_size, ksize1=5, ksize2=9, device='cuda'):
    """
    Generate a mask with 1 at the point with maximum Difference of Gaussians (DoG) response in each tile,
    and return the intensity mask of the max response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        ksize1 (int): Size of the first Gaussian kernel (smaller sigma).
        ksize2 (int): Size of the second Gaussian kernel (larger sigma).
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max DoG response points, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max DoG response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)

    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Apply Gaussian blur with two different kernel sizes
            blur1 = cv2.GaussianBlur(tile, (ksize1, ksize1), 0)
            blur2 = cv2.GaussianBlur(tile, (ksize2, ksize2), 0)
            
            # Compute DoG
            dog = blur1 - blur2
            
            # Find the point with maximum absolute response
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.abs(dog))
            
            # Choose the point with maximum absolute response
            if np.abs(min_val) > max_val:
                x, y = min_loc
                intensity = np.abs(min_val)
            else:
                x, y = max_loc
                intensity = max_val
                
            x = left + x
            y = top + y
            if 0 <= x < W and 0 <= y < H:
                mask_np[y, x] = True
                intensity_np[y, x] = intensity
            else:
                # Fallback to center if out of bounds
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # Use the center's DoG response as intensity
                center_in_tile_y = center_y - top
                center_in_tile_x = center_x - left
                if 0 <= center_in_tile_y < tile_h and 0 <= center_in_tile_x < tile_w:
                    intensity_np[center_y, center_x] = np.abs(dog[center_in_tile_y, center_in_tile_x])
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_eig_max_response_mask(image_tensor, tile_size, block_size=2, ksize=3, device='cuda'):
    """
    Divide an image into tiles, compute Shi–Tomasi (min eigenvalue) responses,
    and generate a mask marking the point with the highest min-eigenvalue in each tile.

    Args:
        image (np.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
        tile_size (tuple): (tile_h, tile_w) specifying the size of each tile.
        block_size (int): Neighborhood size for cornerMinEigenVal.
        ksize (int): Aperture size for Sobel operator.

    Returns:
        mask (np.ndarray): Boolean mask of shape (H, W) with True at selected corners.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Compute min eigenvalue (Shi–Tomasi) response
            eig = cv2.cornerMinEigenVal(tile, block_size, ksize)
            
            # Clamp negative values (floating-point errors) to zero
            eig = np.clip(eig, a_min=0, a_max=None)

            # Find the location of the maximum response in this tile
            _, max_val, _, max_loc = cv2.minMaxLoc(eig)            
            
            x = left + max_loc[0]
            y = top + max_loc[1]
            mask_np[y, x] = True
            intensity_np[y, x] = max_val
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_harris_max_response_mask(image_tensor, tile_size, block_size=2, ksize=3, k=0.04, device='cuda'):
    """
    Generate a mask with 1 at the Harris corner with maximum response in each tile,
    and return the intensity mask of the max Harris response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter for Sobel operator.
        k (float): Harris detector free parameter.
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max Harris corners, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max Harris response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('float32')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Compute Harris response
            harris = cv2.cornerHarris(tile, block_size, ksize, k)
            
            # Find the point with maximum response
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris)
            
            if max_val > 0:  # Only consider if we found a corner
                x = left + max_loc[0]
                y = top + max_loc[1]
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_val
            else:
                # Fallback to center if no corners detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # Use the center's Harris response as intensity (if available)
                center_in_tile_y = center_y - top
                center_in_tile_x = center_x - left
                if 0 <= center_in_tile_y < tile_h and 0 <= center_in_tile_x < tile_w:
                    intensity_np[center_y, center_x] = harris[center_in_tile_y, center_in_tile_x]
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_orb_max_response_mask(image_tensor, tile_size, max_features=1, device='cuda'):
    """
    Generate a mask with 1 at the ORB feature point with maximum response in each tile,
    and return the intensity mask of the max ORB response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        max_features (int): Maximum number of features to detect (we'll use top 1).
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max response ORB features, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max ORB response values at mask points, 0 elsewhere.
    """
    image_np = image_tensor.cpu().numpy()
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('uint8')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(max_features)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Detect ORB features in the tile
            keypoints = orb.detect(tile, None)
            
            if len(keypoints) > 0:
                # Find the keypoint with maximum response
                max_kp = max(keypoints, key=lambda kp: kp.response)
                x = left + int(max_kp.pt[0])
                y = top + int(max_kp.pt[1])
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_kp.response
            else:
                # Fallback to center if no features detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # For center fallback, intensity is set to 0 (no feature response)
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_fast_max_response_mask(image_tensor, tile_size, fast_threshold=10, device='cuda'):
    """
    Generate a mask with 1 at the FAST feature point with maximum response in each tile,
    and return the intensity mask of the max FAST response values.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor on GPU with shape (C, H, W).
        tile_size (tuple): (tile_h, tile_w) size of each tile.
        fast_threshold (int): Threshold for FAST feature detector.
        device (str): Device to put the output tensors on ('cuda' or 'cpu').
    
    Returns:
        mask (torch.Tensor): Boolean mask tensor of shape (H, W),
                            with 1 at max response FAST features, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with max FAST response values at mask points, 0 elsewhere.
    """
    # Convert torch tensor to numpy array on CPU
    image_np = image_tensor.cpu().numpy()
    # Convert to HWC format and grayscale for OpenCV
    if image_np.shape[0] == 3:  # C,H,W -> H,W,C
        image_np = np.transpose(image_np, (1, 2, 0))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('uint8')
    
    H, W = gray.shape[:2]
    tile_h, tile_w = tile_size
    
    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)
    
    # Initialize FAST detector with nonmax suppression disabled
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, 
                                          nonmaxSuppression=True,
                                          type=cv2.FastFeatureDetector_TYPE_5_8)
    # fast.setNonmaxSuppression(False)
    
    # Process each tile
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            
            # Extract tile
            tile = gray[top:bottom, left:right]
            
            # Detect FAST features in the tile with response scores
            keypoints = fast.detect(tile, None)
            
            if len(keypoints) > 0: 
                # Find the keypoint with maximum response
                max_kp = max(keypoints, key=lambda kp: kp.response)
                x = left + int(max_kp.pt[0])
                y = top + int(max_kp.pt[1])
                if 0 <= x < W and 0 <= y < H:
                    mask_np[y, x] = True
                    intensity_np[y, x] = max_kp.response
            else:
                # Fallback to center if no features detected
                center_y = (top + bottom - 1) // 2
                center_x = (left + right - 1) // 2
                mask_np[center_y, center_x] = True
                # For center fallback, intensity is set to 0 (no feature response)
    
    # Convert back to torch tensor and move to specified device
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask

def generate_tile_center_mask(image_shape, tile_size, device='cuda'):
    """
    Generate a mask with 1 at the center of each tile over the image,
    and return an intensity mask with 1 at tile centers (since center has no specific intensity).
    
    Args:
        image_shape (tuple): (H, W) size of the image.
        tile_size (tuple): (tile_h, tile_w) size of each tile.
    
    Returns:
        mask (np.ndarray): Boolean/integer mask of shape (H, W),
                           with 1 at tile centers, 0 elsewhere.
        intensity_mask (torch.Tensor): Float tensor of shape (H, W),
                                     with 1 at tile centers, 0 elsewhere.
    """
    if len(image_shape) == 3 and image_shape[0] == 3:
        H, W = image_shape[1:]
    else:
        H, W = image_shape[:2]
    tile_h, tile_w = tile_size

    mask_np = np.zeros((H, W), dtype=bool)
    intensity_np = np.zeros((H, W), dtype=np.float32)

    # Calculate center positions of tiles
    for top in range(0, H, tile_h):
        for left in range(0, W, tile_w):
            bottom = min(top + tile_h, H)
            right = min(left + tile_w, W)
            center_y = (top + bottom - 1) // 2
            center_x = (left + right - 1) // 2
            mask_np[center_y, center_x] = True
            intensity_np[center_y, center_x] = 1.0  # Center intensity set to 1
    
    mask = torch.from_numpy(mask_np).to(device)
    intensity_mask = torch.from_numpy(intensity_np).to(device)
    return mask, intensity_mask