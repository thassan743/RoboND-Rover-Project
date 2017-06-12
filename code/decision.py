import numpy as np

#-- Function to control the speed of the rover about the desired speed at given throttle
def speed_control(Rover, speed, throttle):
    if Rover.vel > speed + 0.1:
        Rover.throttle = 0
        Rover.brake = Rover.brake_nom
    elif Rover.vel < speed:
        Rover.throttle = throttle
        Rover.brake = 0
    else:
        Rover.throttle = 0
        Rover.brake = 0

#-- Function to stop the rover at desired brake value
def rover_stop(Rover, brake_val):
    Rover.throttle = 0
    Rover.brake = brake_val
    Rover.steer = 0

#-- function to determine which direction to steer to get to the target
#-- ccw is positive steer, cw is negative steer
def steer_dirn(Rover):
    if ((Rover.yaw_error + 360) % 360) < 180:
        return 1
    else:
        return -1

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
#-- Possible Modes:
#-- start: Initial state that the rover starts in. Allows for saving of start location
#-- forward: Rover is moving forward
#-- stop: Rover is stopping or has stopped
#-- vis_target: Rock sample target is visible in image and Rover is navigating to sample
#-- pickup: Rover is at the sample (Rover.near_sample == True). Wait for rover to pick up sample
#-- stuck: Rover is stuck (hasn't moved for a certain amount of time in 'forward' mode)
#-- go_home: All 6 rock samples found, navigate back to start location
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Check for Rover.mode status
    # Rover
    #-- Start mode used to save initial position.
    #-- Point rover in ideal start heading then transition to 'forward'
    if Rover.mode == 'start':
        Rover.target_yaw = 170
        Rover.yaw_error = (Rover.target_yaw - Rover.yaw)
        #-- Save home position
        if Rover.home is None:
            Rover.home = Rover.pos
        #-- turn rover to face ideal start heading
        elif np.absolute(Rover.yaw_error) > 5:
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = 15 * steer_dirn(Rover)
        else:
            Rover.steer = 0
            Rover.mode = 'forward'
    
    #-- Normal driving state of the rover.
    elif Rover.mode == 'forward': 
        #-- Crude check to determine if the rover is stuck
        #-- If the rover is in 'forward' mode but the velocity stays < 0.1 for
        #-- more than 50 cycles, assume stuck
        if (Rover.vel < 0.1) & (Rover.throttle > 0):
            Rover.count += 1
        else:
            Rover.count = 0
            
        if Rover.count >= 70:
            Rover.mode = 'stuck'
            rover_stop(Rover, Rover.brake_set)
            
        elif len(Rover.nav_angles) >= Rover.stop_forward:
            # If mode is forward, navigable terrain looks good 
            # and velocity is below max, then throttle 
            if Rover.vel < Rover.max_vel:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set
            else: # Else coast
                Rover.throttle = 0
            Rover.brake = 0
            
            #-- Check for obstacles directly in front of rover.
            #-- This helps to determine if rover is stuck behind an obstacle
            if (len(Rover.obs_dists[Rover.obs_dists < 10]) > 1):
                Rover.obs_stuck = True
            else:
                Rover.obs_stuck = False
            
            #-- Determine angle to steer.
            #-- Only consider the area directly in front of the rover.
            angles = Rover.nav_angles[(Rover.nav_dists < Rover.nav_close)]
            #-- catch error when no terrain pixels in image
            #-- offset steering angle by x degrees to make the rover hug the wall.
            if (len(angles) > 0):
                Rover.steer = np.clip((np.mean(angles * 180/np.pi) - Rover.nav_adjust), -15, 15)
            
            #-- If a rock is visible in the image, transition to 'vis_target' mode
            #-- Save current yaw heading to continue in the same direction after picking up the sample
            #-- This helps to ensure most of the map is explored
            if (len(Rover.rock_angles) > 0):
                rock_angle = np.mean(Rover.rock_angles * 180/np.pi)
                if rock_angle < 35:
                    rover_stop(Rover, Rover.brake_set)
                    if Rover.vel == 0:
                        Rover.mode = 'vis_target'
                        Rover.target_yaw = Rover.yaw
                        Rover.count = 0
            
            #-- Check if all rock samples have been collected. If so, calculate the distance to home position
            #-- This is done here to allow the 'forward' mode to be used even when all samples have been collected
            #-- Only if the rover is within proximity of the home location, transition to 'go_home' state
            if (Rover.sample_count >= 6):
                Rover.dist_home = np.sqrt((Rover.pos[0]-Rover.home[0])**2 + (Rover.pos[1]-Rover.home[1])**2)
                if (Rover.dist_home < Rover.home_prox):
                    Rover.mode = 'go_home'    
        
        # If there's a lack of navigable terrain pixels then go to 'stop' mode
        elif len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                # Set brake to stored brake value
                rover_stop(Rover, Rover.brake_set)
                Rover.mode = 'stop'

    # If we're already in "stop" mode then make different decisions
    elif Rover.mode == 'stop':
        #-- reset stuck counter
        Rover.count = 0
        # If we're in stop mode but still moving keep braking
        if Rover.vel > 0.2:
            rover_stop(Rover, Rover.brake_nom)
            
        # If we're not moving (vel < 0.2) then do something else
        elif Rover.vel <= 0.2:
            # Now we're stopped and we have vision data to see if there's a path forward
            if len(Rover.nav_angles) < Rover.go_forward:
                Rover.throttle = 0
                # Release the brake to allow turning
                Rover.brake = 0
                # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                Rover.steer = 15 # Turn left since the rover is following the right wall
            # If we're stopped but see sufficient navigable terrain in front then go!
            if len(Rover.nav_angles) >= Rover.go_forward:
                # Set throttle back to stored value
                Rover.throttle = Rover.throttle_set
                # Release the brake
                Rover.brake = 0
                # Set steer to mean angle
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                Rover.mode = 'forward'
    
    #-- In this mode, the sample is visible in the Rover's vision image and it is driving towards the sample
    elif Rover.mode == 'vis_target':
        
        #-- If sample is still visible, navigate towards it
        if (len(Rover.rock_angles) > 0):
            Rover.target_angle = np.mean(Rover.rock_angles * 180/np.pi)
            #-- If sample is more than 15 degrees away, 4 wheel turn
            #-- Otherwise drive to sample
            if np.absolute(Rover.target_angle) > 15:
                if Rover.vel > 0:
                    rover_stop(Rover, Rover.brake_set)
                else:
                    Rover.brake = 0
                    Rover.yaw_error = Rover.target_angle
                    Rover.steer = 15 * steer_dirn(Rover)
            else:
                speed_control(Rover, 0.5, 0.3)
                Rover.steer = Rover.target_angle
            #-- Check if rover gets stuck while trying to drive to a sample
            if (Rover.throttle > 0) & (Rover.vel < 0.1):
                Rover.count += 1
            else:
                Rover.count = 0
            #-- If stuck, hit the throttle. This usually occurs when sample is on a bend    
            if Rover.count >= 150:
                Rover.steer = 0
                Rover.throttle = 10
                Rover.count = 0
        #-- If sample is no longer visible for 20 cycles then go back to 'forward' state
        #-- Prevents the rover from driving away if it temporarily loses sight of sample
        else:
            Rover.count1 += 1
            if Rover.count1 > 20:
                Rover.count1 = 0
                Rover.mode = 'forward'
        #-- If in range to pick up sample, hit the brakes and transition to 'pickup' state
        if (Rover.near_sample):
            rover_stop(Rover, Rover.brake_set)
            Rover.mode = 'pickup'
            Rover.count1 = 0
    
    #-- In pickup state, stop and wait for rover to pick up sample. This is done automatically    
    elif Rover.mode == 'pickup':
        if Rover.near_sample:
            rover_stop(Rover, Rover.brake_set)
        #-- Once sample is picked up, near_sample flag is cleared. Rover turns back to original heading
        #-- Once back to original heading, go back to 'forward' state
        else:
            Rover.brake = 0
            Rover.yaw_error = Rover.target_yaw - Rover.yaw
            if np.absolute(Rover.yaw_error) > 5:
                Rover.steer = 15 * steer_dirn(Rover)
            else:
                Rover.sample_count += 1
                Rover.mode = 'forward'
    
    #-- In this mode, the rover has determined that it is stuck.
    #-- Procedure to free itself could be improved significantly.
    #-- However, this seems to work for most cases even if it takes a bit longer
    elif Rover.mode == 'stuck':
        #-- Check if 'stuck' flag not set (first stuck cycle)
        #-- determine best direction to steer out of stuck position (towards most navigable terrain)
        if not Rover.stuck:
            Rover.stuck = True
            Rover.count = 0
            Rover.throttle = 0
            if np.sign(np.mean(Rover.nav_angles * 180/np.pi)) < 0:
                Rover.yaw_error = -1
            else:
                Rover.yaw_error = 1
        #-- if stuck directly behind a rock, turn ccw and for longer
        if Rover.obs_stuck:
            max_count = 50
            Rover.obs_stuck = False
            Rover.yaw_error = 1
        else:
            max_count = 20
        #-- 4 wheel steer towards terrain for 20 cycles then go back to 'forward' mode and try to drive
        if Rover.count < max_count:
            Rover.steer = Rover.yaw_error*15
            Rover.brake = 0
            Rover.throttle = 0
            Rover.count += 1
        #-- If going home, use a different procedure to account for the small rock obstacles close to home location
        elif Rover.stuck_home:
            speed_control(Rover, 0.5, 0.5)
            Rover.count +=1
            if Rover.count > 120:
                Rover.stuck_home = False
                rover_stop(Rover, Rover.brake_nom)
                Rover.count = 0
                Rover.stuck = False
                Rover.mode = 'go_home'
        else:
            Rover.count = 0
            rover_stop(Rover, Rover.brake_set)
            Rover.stuck_obs = False
            Rover.stuck = False
            Rover.mode = 'forward'
    
    #-- In this mode, the rover returns to the start position.
    #-- Entered once all the samples have been collected and the rover is in close proximity of home
    elif Rover.mode == 'go_home':
        
        #-- Update distance to home and calculate direction to home
        Rover.dist_home = np.sqrt((Rover.pos[0]-Rover.home[0])**2 + (Rover.pos[1]-Rover.home[1])**2)
        Rover.target_yaw = (np.arctan2(Rover.home[1] - Rover.pos[1], Rover.home[0] - Rover.pos[0]) *180/np.pi) %360
        Rover.yaw_error = (Rover.target_yaw - Rover.yaw)# % 360
        
        #-- not home yet
        if Rover.dist_home > 1:
            
            #Rover.ter_angle = np.mean(Rover.nav_angles * 180/np.pi)
            
            #-- If the rover is not facing home, stop and 4 wheel steer to face home.
            #-- This prevents driving in circles around home position
            if np.absolute(Rover.yaw_error) > 10:
                if (Rover.vel > 0):
                    rover_stop(Rover, Rover.brake_set)
                else:
                    Rover.brake = 0
                    Rover.throttle = 0
                    Rover.steer = 15 * steer_dirn(Rover)
            #-- If facing home, drive towards home position
            else:            
                speed_control(Rover, 1.0, 0.2)
                Rover.steer = np.absolute(Rover.yaw_error) * steer_dirn(Rover)
                #-- check if stuck
                if Rover.vel < 0.1:
                    Rover.count += 1
                else:
                    Rover.count = 0
                    
                if Rover.count >= 50:
                    Rover.mode = 'stuck'
                    Rover.stuck_home = True
                    rover_stop(Rover, Rover.brake_set)
                            
        #-- Once home, stop
        else:
            rover_stop(Rover, Rover.brake_set)
            
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

