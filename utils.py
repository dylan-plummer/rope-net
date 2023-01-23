import os
import base64 
import shutil

def remove_directory_containing_file(file_path):
   directory_path = os.path.dirname(file_path)
   try:
      shutil.rmtree(directory_path)
   except Exception as e:
      print(e)


def base64_encode_video(video_file, alt=""):
   # http://www.iandevlin.com/blog/2012/09/html5/html5-media-and-data-uri/
   with open(video_file, "rb") as file:
      text = base64.b64encode(file.read())
   decoded = text.decode('ascii')
   return str(f"data:video/webm;base64,{decoded}")


def progress_load_model():
   return None, None, {'display': 'inline-block'}, "Loading model...", True, False, ""


def progress_inference(i, total, msg, count_msg=""):
   return str(i), str(total), {'display': 'inline-block'}, msg, True, False, count_msg
