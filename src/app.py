import requests
import streamlit as st
import tkinter as tk

from tkinter import filedialog
from pathlib import Path


root = tk.Tk()
root.withdraw()
# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

endpoint_url = 'http://localhost:8000/ajax'


@st.cache(allow_output_mutation=True)
def get_state():
    return {
        "council": ""
    }


state = get_state()
council = state.get("council", None)


def clean_council(text):
    return text.replace(" ", "_")


# Folder picker button
st.title('Upload Images and Run Detection')

council = st.text_input("Add council name", value=council)  # , on_change=check_values)
year = st.text_input("Year Images Taken")

project_name = f"RACAS_{clean_council(council)}_{year}"

project_identifier = st.text_input("Project Identifier", project_name)

st.write('Please select a folder:')
clicked = st.button('Folder Picker')

if clicked:
    if council == "":
        st.warning("Please enter council name.")
    elif year == "":
        st.warning("Please enter road survey year")
    elif project_identifier == "":
        st.warning("Please enter project identifier (or edit year twice)")
    else:  # Only process if all of the above checks passed.
        dir_name = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
        dst_folder = Path.cwd() / "tempDir" / Path(dir_name).name
        dst_folder.mkdir(parents=True, exist_ok=True)
        st.write(f"Made {str(dst_folder)}")
        for path in Path(dir_name).iterdir():
            if not path.suffix.lower() == ".jpg":
                continue
            with open(str(path), "rb") as jpg_img:
                files = {'image': (path.name, jpg_img.read(), 'multipart/form-data', {'Expires': '0'})}  # b64_string
                resp = requests.post(endpoint_url, files=files, data={"project_ref": project_identifier}),
            # get detections.ai file and accumulate the result. The accumulated file will help build marks, POI
            # when building a shape file for the project
            print(f"Detections in {path.name}: ", resp[0].json()["detections"])  # Replace key msg1 with "detections"

# @st.cache
# def process_image(image):
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
#     )
#     net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
#     net.setInput(blob)
#     detections = net.forward()
#     return detections


# @st.cache
# def annotate_image(
#         image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
# ):
#     # loop over the detections
#     (h, w) = image.shape[:2]
#     labels = []
#     for i in np.arange(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#
#         if confidence > confidence_threshold:
#             # extract the index of the class label from the `detections`,
#             # then compute the (x, y)-coordinates of the bounding box for
#             # the object
#             idx = int(detections[0, 0, i, 1])
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             # display the prediction
#             label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
#             labels.append(label)
#             cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             cv2.putText(
#                 image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
#             )
#     return image, labels

# st.title("Object detection with MobileNet SSD")
# img_file_buffer = st.file_uploader("Upload an image", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
# confidence_threshold = st.slider(
#     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
# )
#
# if img_file_buffer is not None:
#     image = np.array(Image.open(img_file_buffer))
#
# else:
#     demo_image = DEMO_IMAGE
#     image = np.array(Image.open(demo_image))
#
# detections = process_image(image)
# image, labels = annotate_image(image, detections, confidence_threshold)
#
# st.image(
#     image, caption=f"Processed image", use_column_width=True,
# )

# st.write(labels)
