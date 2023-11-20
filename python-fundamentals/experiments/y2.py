# importing the module
from pytube import YouTube

# download location of the file
DOWNLOAD_PATH = " C:/Users/Jobhunt/Downloads/"
# link of the video to be downloaded
link = "https://youtu.be/oQdxL_WW3aE"
youtube_obj = YouTube(link)

# filters out all the files with "mp4" extension
mp4files = youtube_obj.streams.filter('mp4')
# to set the name of the file
youtube_obj.set_filename('Downloaded Video')
# get the video with the extension and
# resolution passed in the get() function
d_video = youtube_obj.get(mp4files[-1].extension, mp4files[-1].resolution)
d_video.download(DOWNLOAD_PATH)
print('Task is Completed!')
