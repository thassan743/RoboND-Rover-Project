import numpy as np
import cv2

#--- Modified to allow for min and max thresholds.
#--- This allows the same function to be used for terrain, obstacles and rock samples
#--- Thresholds are 2 tuples for min and max: (minR,minG,minB), (maxR,maxG,maxB)
# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, thresh_min = (0,0,0), thresh_max = (255,255,255), operator = 'and'):
    # Create an array of zeros same xy size as img, but single channel
    select = np.zeros_like(img[:,:,0])
    
    #-- Require that each pixel be above all three min and below all three max threshold values in RGB
    #-- above_thresh will now contain a boolean array with "True" where threshold was met
    #-- allow for both 'AND' or 'OR' operators.
    #-- OR seems to work better for obstacles and AND for terrain and rocks
    #-- Threshold array for navigable terrain. 160 works well for min RGB
    if (operator == 'and'):
        above_thresh = ((img[:,:,0] > thresh_min[0]) & (img[:,:,0] <= thresh_max[0])) \
                     & ((img[:,:,1] > thresh_min[1]) & (img[:,:,1] <= thresh_max[1])) \
                     & ((img[:,:,2] > thresh_min[2]) & (img[:,:,2] <= thresh_max[2]))
    elif (operator == 'or'):
        above_thresh = ((img[:,:,0] > thresh_min[0]) & (img[:,:,0] <= thresh_max[0])) \
                     | ((img[:,:,1] > thresh_min[1]) & (img[:,:,1] <= thresh_max[1])) \
                     | ((img[:,:,2] > thresh_min[2]) & (img[:,:,2] <= thresh_max[2]))
    
    # Index the array of zeros with the boolean array and set to 1
    select[above_thresh] = 1
    
    # Return binary images
    return select
    
# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
#--- Modified to complete rotation function
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
#--- Modified to complete translation function
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated
    
# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world
    
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    
    # 1) Define source and destination points for perspective transform
    
    # Define calibration box in source (actual) and destination (desired) coordinates
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    #-- Min and Max threshold's for navigable terrain, obstacles, and rock samples
    nav_min = (160, 160, 160)
    nav_max = (255,255,255)
    obs_min = (5, 5, 5)
    obs_max = (70,70,70)
    rock_min = (170, 130, 0)
    rock_max = (255, 190, 60)
    
    #-- Call modified color_thresh function on the warped image.
    #-- Returns threshold images for terrain, obstacles and rock samples
    thresh_nav = color_thresh(warped, nav_min, nav_max, 'and')
    thresh_obs = color_thresh(warped, obs_min, obs_max, 'or')
    thresh_rock = color_thresh(warped, rock_min, rock_max, 'and')
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = thresh_obs * 255
    Rover.vision_image[:,:,1] = thresh_rock * 255
    Rover.vision_image[:,:,2] = thresh_nav * 255
        
    # 5) Convert map image pixel values to rover-centric coords
    
    # Extract navigable terrain pixels in rover-centric frame
    xpix_nav, ypix_nav = rover_coords(thresh_nav)
    # Extract obstacle pixels in rover-centric frame
    xpix_obs, ypix_obs = rover_coords(thresh_obs)
    # Extract rock pixels in rover-centric frame
    xpix_rock, ypix_rock = rover_coords(thresh_rock)
    
    # 6) Convert rover-centric pixel values to world coordinates
    
    # scale since 1 pixel = 1m in map but 0.1m in image
    scale = 10
    # Convert terrain, obstacles and rock sample pixels to world coord frame.
    # Pass rover x, y, yaw from data class
    # Get navigable terrain pixel positions in world coords
    navigable_x_world, navigable_y_world = pix_to_world(xpix_nav, ypix_nav,
                                Rover.pos[0], Rover.pos[1], Rover.yaw, 
                                Rover.worldmap.shape[0], scale)
    # Get obstacle pixel positions in world coords
    obs_x_world, obs_y_world = pix_to_world(xpix_obs, ypix_obs,
                                Rover.pos[0], Rover.pos[1], Rover.yaw, 
                                Rover.worldmap.shape[0], scale)
    # Get rock pixel positions in world coords
    rock_x_world, rock_y_world = pix_to_world(xpix_rock, ypix_rock,
                                Rover.pos[0], Rover.pos[1], Rover.yaw, 
                                Rover.worldmap.shape[0], scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    
    #-- To optimize map fidelity, only update map when pitch and roll are near zero
    if (((Rover.roll < Rover.roll_max) | (Rover.roll > 360-Rover.roll_max))
      & ((Rover.pitch < Rover.pitch_max) | (Rover.pitch > 360-Rover.pitch_max))):
        #-- Add obstacles to world map on RED layer
        Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
        #-- Add rock samples to world map on GREEN layer. Increase weight to make locations visible
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        #-- Add navigable terrain to world map on BLUE layer
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_nav, ypix_nav)
    Rover.rock_dists, Rover.rock_angles = to_polar_coords(xpix_rock, ypix_rock)
    Rover.obs_dists, Rover.obs_angles = to_polar_coords(xpix_obs, ypix_obs)
    
    return Rover