## Project: Search and Sample Return
### Taariq Hassan (thassan743)

---

[//]: # (Image References)

[image1]: ./misc/thresh.png
[image2]: ./misc/mosaic.png
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

In the notebook the `color_thresh()` function was modified to allow for upper and lower threshold arguments. This allowed the same function to be used for applying thresholds for terrain, obstacles, and rock samples, preventing having to write new functions to do almost the same thing.

The upper and lower threshold values used for terrain, obstacles, and rocks were determined experimentally using an interactive `matplotlib` window earlier in the notebook. Images were obtained from the provided `test_dataset` as well as from data that was recorded in `my_dataset`.

A further addition to the `color_thresh()` function was adding an operator argument. This allowed for either the AND (`&`) or OR (`|`) operator to be used between thresholds of different colours. Thresholding the terrain and rock samples were found to work well using the AND operator, but for obstacles, the OR operator produced better results.

The image below shows the results of applying each of the thresholds to a single image. The difference between terrain, obstacles, and rocks can clearly be seen.

![alt text][image1]
#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

The `process_image()` function was updated with the upper and lower thresholds for terrain, obstacles and rocks. This was followed by three calls to `color_thresh()`, passing each pair of thresholds one at a time to generate the three thresholded images. These images were then transformed to the rover-cetric and the world reference frames respectively.

The mosaic image contained the original image from the rover in the top left corner.

In the top right corner is the rover's vision image which is all three threshold images combined and assigned to appropriate colour channels. This was done by creating a new 3 channel image and assigning the obstacles, rock, and terrain binary images to the R, G and B channels respectively. Since the threshold images were binary, their values were multiplied by 255 when adding it to the vision image.

In the bottom right corner is the world map that the rover generates. This was created in a similar way to the vision image above but using the world reference frame transformed images. Channel allocation was the same as the vision image.

Finally, the bottom left image contains the same world map but overlayed on the ground truth map of the environment

One problem that was encountered was that in most cases, the rover sees obstacles across the top of its vision image. This means that as it's driving, obstacles are detected almost everywhere. This has the effect of making the terrain look purple in most places on the world map since the blue and red channels are both 255 at those points. A solution to get around this may be to only update the world map with pixels closest to the rover where the terrain is well defined. However, this issue was not present when running the same code in the simulator and was therefore left as is.

One thing whcih I did not like was that the rock samples display on the world map as large yellow streaks rather than just points (see the screenshot below). This is as a result of how the perspective transform works. With some effort, it would be possible to get the samples to display better. Again, however, this issue was not apparent in the simulator since the provided `create_output_images()` function in `supporting_functions.py` takes care of it for us. It was therefore also left as is.

The image below is a screenshot from the video generated by `moviepy` on the recorded data from `my_dataset`. The video can be found in `my_dataset/output`.
**Note: the recorded images were not added to the repository since there were just too many files.**

![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

#### `perception_step()`

For the most part, the `perception_step()` function is the same as the `process_image()` function from the notebook. First, the perspective transform is applied to the source image from the rover. Then the `color_thresh()` function is called to get the threshold images for terrain, obstacles, and rocks. These images are then used to update the rover vision image, as well as being converted to rover-centric and then to world co-ordinates.

Now that the images have been converted to the world reference frame, the world map can be updated. However, due to the nature of the perspective transform, if the source image is not perfectly aligned with the environment, the transformed images become extremely inaccurate when converted to world co-ordinates, thereby reducing the fidelity of the generated map. In the case of the rover, as it accelerates, brakes, or drives over the uneven terrain, it rolls and pitches about it's own axis. Therefore, to increase fidelity, the world map is only updated when the rovers roll and pitch are close to zero. The maximum allowable threshold is experimental and there is a tradde-off between accuracy(fidelity) and the percentage of the environment that will be mapped. Eventually, a maximum allowable pitch of 0.2 degrees and roll of 0.5 degrees was chosen.

Finally, the angles and distances to terrain pixels were determined in the rover-centric reference frame using the `to_polar_coords()` function. Additionally, the angles and distances to obstacle and rock pixels were also calculated, which were used lated in the `decision_step()`. 

#### `decision_step()`

In this function lies the bulk of the rovers "intelligence". At any given time, the rover exists in one of the possible modes. Each mode is handled using `if-elif-else` blocks of code. The different rover modes are described in more detail below.

##### start mode
Quite simply, this is the initial state of the rover. In this state, the rover saves its starting location such that it can return later. The rover also turns to a predetermined heading such that each run was more repeatable.

##### forward mode 
This is the main driving mode of the rover.
The first thing the rover does is check if it is stuck. This is determined checking whether velocity is less than 0 and throttle is greater than 0, which means the rover is trying to drive but can't move. If this occurs for a set amount of time, the rover transitions to the `stuck` mode.

If the rover is not stuck, it proceeds to determne if there is enough navigable terrain in front of it. If there is, the rover drives, regulating its speed about the desired setpoint. The angle at which to steer is calculated by determining the average of the angles to all navigable terrain pixels that the rover can see. However, it was found that with this method, the rover would not always explore small areas of the environment where there were few navigable terrain pixels, and would sometimes drive around in circles in areas with large amount of navigable terrain. Therefore, to improve performance and increase the probability of the rover exploring more of the environment, only the terrain within 40m of the rover was used to calculate the steering angle. To further improve the explored area, the rover was made to follow the right wall. This was done by subtracting a fixed offset from the calculated steering angle.

The rover then checks to see if any rock samples are visible. If a sample is visible, the rover stores its current heading and then transitions to the `vis_target` mode. Since the rover was made to follow the right wall, it was found that in some cases when a sample was seen close to the left wall, the rover would pick up the sample then turn around and go back from where it came instead of continuing forward and exploring the rest of the area. For this reason, the current heading was stored, and samples greater than a certain angle from the rover were ignored since they would be seen when the rover returns along the opposite wall.

The rover also then checks to see if it has collected all the samples and if it is close to the home position, in which case it would transition to the `go_home` mode.

Finally, if there weren't enough navigable terrain pixels in front of the rover, it transitions to the `stop` mode.

##### stop mode
In this mode, the rover is either stationary or coming to a stop. If the rover has stopped, it checks to see how much navigable terrain is visible in front of it. If there is sufficient terrain, the rover transitions to the `forward` mode. If not, the rover does a 4-wheel turn counter-clockwise until there is sufficient visible terrain. Counter-clockwise was chosen since the rover follows the right wall.

##### vis_target mode
In this mode, a rock sample is visible in the rover's vision image. The rover calculates the angle to the sample and steers towards it. Some checks are done to determine if the rover gets stuck while tring to get to the sample. A counter is also used to prevent the rover from driving away if it temporarily loses sight of the sample. Once the rover reaches the sample (`near_sample` flag set) the rover stops and transitions to the `pickup` mode. Alternatively, if the rover has lost sight of the sample, it transitions to the `forward` mode.

##### pickup mode
The rover automatically picks up samples if it is stationary and close enough to the sample (`near_sample` flag set). Therefore, in this state the rover stops until the sample has been picked up, at which point it 4-wheel turns back to its original heading and transitions to the `forward` mode.

##### stuck mode
If the rover is in this mode, it has determined that it is stuck. The process of freeing itself is not very efficient and could use some work. However, in most cases the rover is able to break free after some time. The direction of the most navigable terrain is calculated and the rover attempts to 4-wheel steer in that direction. For cases where the rover is stuck along a wall, this method works reasonable well. However, if the rover is stuck behind the small or large rock obstacles, the method is far less effective. In this case, the rover tries to determine if there is an obstacle directly in front of it and then 4-wheel steer for longer so as to point away from it. The rover then transitions to the `forward` mode to determine if it is free.

A speacial case was also created for when the rover is attempting to drive home and gets stuck on the small rocks close to the centre of the map. In this case, the rover transition to the `go_home` mode.

##### go_home mode
In this mode, the rover simply drives towards the home location. The rover already navigates itself close to the home location in the forward mode once all samples have been collected. Therefore, in this mode, the rover just drives straight for home without considering the terrain. In case it does get stuck, the rover transitions to the `stuck` mode. Otherwise, once the rover is within 1 meter of the home location, it stops.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

The rover was able to successfully complete its mission, mapping **97.3%** of the environment with a fidelity of **76.9%** and collecting all **6** rock samples in approximately **13** minutes. The simulator was run at a resolution of **1280x720** with the quality set to **good** at an average frame rate of **27 FPS**. A recording of the run can be found [here](https://youtu.be/3ypcx4DqvGs)

I am reasonably happy with the performance of the rover, however, there are many areas that can be improved.

The current procedure to free the rover when stuck is far from optimal. In some cases, the rover 4-wheel turns left, then right, then left a number of times with no improvement, eventually freeing itself after some time. A better method could be implemented, however the best option would be to implement better path planning algorithms, such that the rover doesn't get stuck in the first place (or at least not as often). This is something I would like to look at implementing in the future.

The rover is also very slow. At higher speeds, the rover rolls and pitches a lot more which reduces fidelity if no limits are imposed on roll and pitch, or reduces the percentage of the area that is mapped if limits are imposed. There is also a chance that the rover will miss the samples. Ideally, the rover should be able to drive fast and still perform well. I think one method would be to modify the perspective transform to account for the rover's attitude (roll and pitch) in which case, no limits on roll and pitch would need to be imposed.

