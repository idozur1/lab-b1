### Paths - set ahead!
NAMES = [ '2.avi', '3.avi', '4.avi'] # List of Names of File (must be .avi files)
WORKDIR = r"C:\Users\idozu\Documents\GitHub\lab-b1\brownian motion" # File-Path (directory)

# Set the time interval (between frames) by frame-rate
TIME_INTERVAL = 1/4.4

# Threshold
THRESHOLD = 97

# Resizing consts
"""
Default video resolution is set to be 2560X1920 (matches the default setting of the cameras in the lab).
In case this is not the resolution of your video, set the SHOULD_RESIZE to true and set the resizing factor so that
<your_resolution>=RESIZING_FACTOR*(2560, 1920).
That will set all the consts that do with size to match your resolution
"""
SHOULD_RESIZE = False # Should be set to True if one would like to resize the video
RESIZING_FACTOR = 1 # By how much one should resize, if resizing is desireable

BACKWARDS_CHECKING_CONST = 100 # The maximum to check for a match in previous frames

### The next consts should be changed unless problems occur. They change what is considered a particle (min\max size) and what is the distance
### that will set 2 particles as the same one between following frames

# Particle set size (if you don't find particles because they are too big or too small change this)
MIN_USER_PARTICLE_SIZE = 32
MAX_USER_PARTICLE_SIZE = 4800
POINT_MAX_USER_DISTANCE_CONST = 80