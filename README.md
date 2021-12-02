# RaceTracker
This is a proof of concept of a simple line tracker to help blind runners stay in between the lines. 

It was developed in Python using open-cv2.

![demoGIF](https://github.com/Shraneid/RaceTracker/blob/main/demo.gif)

*Note that the jump the blue line does at the end is due to the video looping back to the beginning*

# Main Algorithm Explained
- Create an [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) mask with the parameters from the controls window
- Multiply the input video stream by the mask to retrieve a matrix of Boolean values (the black and white video, bottom right)
- Apply the [Hough Lines](https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html) algorithm to detect lines 
- This gives us the green lines that you can see
- Once only two lines are visible on screen, finish initialization and start tracking those two lines (this could be improved by adding a button of some sort to trigger initialization)
- Now we reapply the same mechanism on every frame and use a custom tracking algorithm to determine which of the newly detected lines correspond to the ones from the previous frame
- Last but not least, we use the position of the tracked lines to determine if we are going straight or if we are deviating on either side, in which case we send a signal (right now printing a message, but could send a beeping sound, or some kind of vibration on either arm of the runner)

# Future Improvements
- The tracking could definitely be improved with simple heuristics :
  - The two middle lines should be the ones moving the less (as the fisheye effect pushes the other further away)
  - Since the camera in kind of fisheye, the two middle lines are the two successive lines that diverge the most
  - Add a threshold on how much rotation a line can support (this might not work if the runner turns the head too much)
  - Have a Subjective mask instead of the Objective one currently being used would allow to diminish the masking issues when the lightness changes during a single take
- Handle the case for corners/turns which is probably not going to work well with our current Hough Lines algorithm ([Hough Circles](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html) with a big enough radius ? And then switch between line and circle seamlessly depending on which is performing best)
- A well trained CNN would probably be much better at this than this naive implementation but would require a lot of data
- Actual 3D positional tracking might help although it might yield uncertain results due to the shaking 

# Author
**Quentin Tourette** - *Initial Work*

**Alix Bouni** - *Medical associate and Project Manager*
