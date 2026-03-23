from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    try:
        audio = video.audio
        if audio is None:
            raise ValueError("video has no audio track")
        audio.write_audiofile(audio_path, verbose=False, logger=None)
    finally:
        video.close()


if __name__ == "__main__":

    extract_audio("test.mp4", "audio.wav")