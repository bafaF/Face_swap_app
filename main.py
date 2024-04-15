import gradio as gr
import os
import shutil
from datetime import datetime
import uuid
import argparse


# Init the project
def init():
    if not os.path.exists("roop"):
        command = (f"git clone https://github.com/based9based/roop && "
                   f"cd roop && "
                   f"pip install -r requirements.txt && "
                   f"wget https://civitai.com/api/download/models/85159 -O inswapper_128.onnx")
        os.system(command)

    # Create folder for the app
    if not os.path.exists("face"):
        os.mkdir("face")
    if not os.path.exists("output"):
        os.mkdir("output")
        os.mkdir("output/folder")
        os.mkdir("output/img")
        os.mkdir("output/video")
    if not os.path.exists("target"):
        os.mkdir("target")
        os.mkdir("target/folder")
        os.mkdir("target/img")
        os.mkdir("target/video")


# Upload function to save the face and video
def upload_file(file, dst_folder=None):
    data = os.path.splitext(file)
    extension = data[1]
    # Separate the files by type
    if extension == ".mp4":
        new_base = "target.mp4"
        if dst_folder is None:
            dst_folder = "target/video/"
    else:
        if dst_folder is None:
            dst_folder = "target/img/"
            new_base = "target.png"
        else:
            new_base = "face.png"

    new_name = os.path.join(dst_folder, new_base)
    shutil.copy(file, new_name)
    return new_name


# Upload the face img
def upload_face(face):
    file_name = face
    dst_folder = "face/"
    return upload_file(file_name, dst_folder)


# Delete all the files in paths (Not the output)
def delete_file(paths):
    for path in paths:
        if path.endswith(".mp4") or path.endswith(".jpg") or path.endswith(".png"):
            os.remove(path)
        else:
            filenames = os.listdir(path)
            for filename in filenames:
                os.remove(os.path.join(path, filename))


# Prepare all the args that will be passed to the swap function
def start_swap(file, face, directory=None, directory_name=None):
    # If there is a directory
    if not directory:
        target_file_path = upload_file(file)
    else:
        target_file_path = file
    # Upload the face img
    face_path = upload_face(face)
    data = os.path.splitext(target_file_path)
    extension = data[1]
    creation_date = str(datetime.now().strftime("%Y-%m-%d"))
    # Put the files in the output folder
    if directory:
        output_folder = "output/folder/" + creation_date + "/" + directory_name
    elif extension == ".mp4":
        output_folder = "output/video/" + creation_date
    else:
        output_folder = "output/img/" + creation_date
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Start swap using the args
    output_file = swap(target_file_path, face_path, output_folder)
    # Return the file or the folder_path
    if not directory:
        paths = [target_file_path, face_path]
        delete_file(paths)
        return output_file
    else:
        return output_folder


# Start the swap using the command in the roop github (https://github.com/based9based/roop)
def swap(target_path, face_path, output_folder, directory=None):
    data = os.path.splitext(target_path)
    extension = data[1]
    # Create an unique filename
    output_filename = str(datetime.now()).replace(" ", "").replace(":", "-").replace(".", "-") + extension
    # Remove the "3" from python if error
    command = f"python3 roop/run.py --target {target_path} --source {face_path} -o {output_folder}/{output_filename} --keep-fps --execution-provider cuda --frame-processor face_swapper face_enhancer"
    os.system(command)
    output_path = output_folder + "/" + output_filename
    if not directory:
        return output_path


# Function of the swap_video_interface
def swap_video(video, face):
    output_video = start_swap(video, face)
    return output_video


# Same function but different name so that gradio won't throw an error
def swap_image(image, face):
    output_image = start_swap(image, face)
    return output_image


# Start swap with the folder
def multiple_swap(face_image, video_path):
    output_folder = ""
    files = os.listdir(video_path)
    directory_name = uuid.uuid4().hex
    for file in files:
        output_folder = start_swap(video_path + file, face_image, True, directory_name)
    return output_folder


# Function of directory_interface
def upload_video_directory(face_image, video_directory):
    files = video_directory
    video_path = "target/folder/"
    # Select all videos and images
    for file in files:
        if file.endswith(".mp4") or file.endswith(".png") or file.endswith(".jpg"):
            # Put each file in the target/folder
            shutil.move(file, video_path)
    output_folder = multiple_swap(face_image, video_path)
    video_path = [video_path, "face/"]
    # Delete all the file used for the swap folder
    delete_file(video_path)
    output_files = os.listdir(output_folder)
    output_files_path = []
    # Return a list of file in output folder path
    for output_file in output_files:
        output_file_path = os.path.join(output_folder, output_file)
        output_files_path.append(output_file_path)
    return output_files_path


# Start the gradio app
def gradio_start(**kwargs):
    # Video_swap_interface inputs and outputs
    video_target_swap = gr.Video(sources=["upload"])
    face_video_image = gr.Image(sources=["upload"], type="filepath")
    swapped_video = gr.Video()

    # Image_swap_interface inputs and outputs
    image_swap = gr.Image(sources=["upload"], type="filepath")
    face_image = gr.Image(sources=["upload"], type="filepath")
    swapped_image = gr.Image()

    video_target_swap.height = 450
    face_video_image.height = 300
    swapped_video.height = 750

    image_swap.height = 450
    face_image.height = 300
    swapped_image.height = 750

    video_interface = gr.Interface(fn=swap_video, inputs=[video_target_swap, face_video_image],
                                   outputs=swapped_video,
                                   allow_flagging="never", submit_btn="Swap")

    image_interface = gr.Interface(fn=swap_image, inputs=[image_swap, face_image], outputs=swapped_image,
                                   allow_flagging="never", submit_btn="Swap")

    face_folder_img = gr.Image(type="filepath", sources=["upload"])
    input_path = gr.File(label="Your videos/img folder", type="filepath", file_count="directory")
    output_path = gr.Files(label="Output videos folder", interactive=False)
    directory_interface = gr.Interface(fn=upload_video_directory, inputs=[face_folder_img, input_path],
                                       outputs=output_path, allow_flagging="never", submit_btn="Swap folder")

    demo = gr.TabbedInterface([video_interface, image_interface, directory_interface],
                              ["Video face swap", "Image face swap", "Directory face swap"])

    # Settings launch args
    launch_kwargs = {}
    server_name = kwargs.get("listen")
    server_port = kwargs.get("server_port", 0)
    launch_kwargs["server_name"] = server_name
    if server_port > 0:
        launch_kwargs["server_port"] = server_port
    launch_kwargs["debug"] = True
    demo.queue()
    demo.launch(**launch_kwargs)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=0,
        help="Port to run the server listener on",
    )
    args = parser.parse_args()
    gradio_start(**vars(args))


if __name__ == "__main__":
    init()
    arg_parser()
