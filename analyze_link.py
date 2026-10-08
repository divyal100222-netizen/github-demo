import logging
import os
import warnings

# 1. MUST BE AT THE VERY TOP (Before importing tensorflow, deepface, or tf_keras)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN warnings
os.environ['TF_KEYPOINT_LOG_LEVEL'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import glob
import shutil
import ssl
import cv2
import deepface
from deepface import DeepFace
import instaloader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 2. SSL Bypass (MUST run before any NLTK downloads/imports that fetch data)
try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
else:
  ssl._create_default_https_context = _create_unverified_https_context

# 3. Ensure Lexicon is Downloaded BEFORE Initializing Analyzer
try:
  sia = SentimentIntensityAnalyzer()
except LookupError:
  nltk.download('vader_lexicon')
  sia = SentimentIntensityAnalyzer()


def analyze_video_frames(video_path, max_frames=5):
  """Extracts key frames across a video and determines overall face emotion."""
  cap = cv2.VideoCapture(video_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if total_frames <= 0:
    print('Could not read video file.')
    return

  step = max(1, total_frames // max_frames)
  emotions = []

  for i in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
      break

    try:
      # DeepFace accepts OpenCV BGR image numpy arrays directly
      results = DeepFace.analyze(
          img_path=frame, actions=['emotion'], enforce_detection=False
      )
      if results and isinstance(results, list):
        emotions.append(results[0]['dominant_emotion'])
    except Exception:
      continue

  cap.release()

  if emotions:
    most_common = max(set(emotions), key=emotions.count)
    print(f'Dominant Video Facial Emotion: {most_common.capitalize()}')
  else:
    print('No face detected in video frames.')


def analyze_instagram_link(post_url):
  # Enable video downloading
  L = instaloader.Instaloader(
      download_videos=True,
      download_video_thumbnails=True,
      download_comments=False,
      save_metadata=False,
  )

  # Extract post shortcode
  shortcode = post_url.split('/p/')[-1].split('/reel/')[-1].split('/')[0]

  print(f'Fetching post data for: {shortcode}...')
  target_dir = f'temp_{shortcode}'

  try:
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    L.download_post(post, target=target_dir)

    # Text Emotion Analysis
    caption = post.caption
    print('\n--- CAPTION ANALYSIS ---')
    if caption:
      print(f'Caption: "{caption[:80]}..."')
      text_scores = sia.polarity_scores(caption)
      compound = text_scores['compound']

      if compound >= 0.05:
        text_emotion = 'Positive / Happy 😊'
      elif compound <= -0.05:
        text_emotion = 'Negative / Sad 😔'
      else:
        text_emotion = 'Neutral 😐'
      print(f'Text Emotion: {text_emotion}')
    else:
      print('No text caption found.')

    # Media Facial Emotion Analysis (Handles both images and videos)
    image_files = glob.glob(f'{target_dir}/*.jpg')
    video_files = glob.glob(f'{target_dir}/*.mp4')

    print('\n--- FACIAL EMOTION ANALYSIS ---')
    if video_files:
      print('Video detected. Sampling frames for facial expressions...')
      analyze_video_frames(video_files[0])
    elif image_files:
      print('Image detected. Analyzing facial expression...')
      try:
        results = DeepFace.analyze(
            img_path=image_files[0],
            actions=['emotion'],
            enforce_detection=False,
        )
        dominant = results[0]['dominant_emotion']
        print(f'Dominant Face Emotion: {dominant.capitalize()}')
      except Exception as e:
        print('Could not process image faces:', e)
    else:
      print('No image or video files found.')

  finally:
    # Cleanup temporary files
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir)


if __name__ == '__main__':
  url = input('Paste Instagram Post URL: ')
  analyze_instagram_link(url)