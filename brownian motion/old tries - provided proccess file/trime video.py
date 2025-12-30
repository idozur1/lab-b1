from moviepy.video.io.VideoFileClip import VideoFileClip

input_video = "2.avi"
output_video = "trimmed_2.avi"
start_time = 0  # Seconds
end_time = 110   # Seconds

# 1. Load the clip
clip = VideoFileClip(input_video)

# 2. Cut it (subclip uses seconds)
new_clip = clip.subclipped(start_time, end_time)

# 3. Save it
# codec='rawvideo' keeps it uncompressed (huge file size but perfect quality)
# codec='libx264' compresses it (standard)
new_clip.write_videofile(output_video, codec='libx264')