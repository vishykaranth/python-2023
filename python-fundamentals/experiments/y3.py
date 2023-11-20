from pytube import YouTube
yt = YouTube("https://www.youtube.com/watch?v=oQdxL_WW3aE")
yt = yt.get('mp4', '720p')
yt.download('C:/Users/Jobhunt/Downloads/')