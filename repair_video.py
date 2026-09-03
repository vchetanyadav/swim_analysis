import imageio_ffmpeg, subprocess
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-err_detect", "ignore_err", "-fflags", "+genpts",
                "-i", "top.mp4", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "top_fixed.mp4"])
print("wrote top_fixed.mp4")