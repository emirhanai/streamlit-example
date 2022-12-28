from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import itertools
import streamlit as st
from PIL import Image
import re
from io import BytesIO
import label
#import pandas as pd
import streamlit_authenticator as stauth


def create_model():
    return SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")


def create_feature_extractor():
    return SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")


def postprocess(masks, height, width):
    masks = F.interpolate(masks, (height, width))

    label_per_pixel = torch.argmax(
        masks.squeeze(), dim=0).detach().numpy()
    color_mask = np.zeros(label_per_pixel.shape + (3,))
    palette = itertools.cycle(sns.color_palette())

    for lbl in np.unique(label_per_pixel):
        color_mask[label_per_pixel == lbl, :] = np.asarray(next(palette)) * 255
    return color_mask, np.unique(label_per_pixel)

def segment(image: Image, model, feature_extractor) -> torch.Tensor:
    inputs = feature_extractor(
        images=image, return_tensors="pt")
    outputs = model(**inputs)
    masks = outputs.logits

    color_mask = postprocess(masks, image.height, image.width)[0]
    labell = postprocess(masks, image.height, image.width)[1]
    pred_img = np.array(image.convert('RGB')) * 0.25 + color_mask * 0.75
    pred_img = pred_img.astype(np.uint8)

    return pred_img,labell



def init():
    st.set_page_config(page_title="Semantic image segmentation")
    st.session_state["model"] = create_model()
    st.session_state["feature_extractor"] = create_feature_extractor()


@st.experimental_memo(show_spinner=False)
def process_file(file):
    return segment(
        Image.open(file),
        st.session_state["model"],
        st.session_state["feature_extractor"]
    )[0]


def get_uploaded_file():
    return st.file_uploader(
        label="Choose a file",
        type=["png", "jpg", "jpeg"],
    )


def download_button(file, name, format):
    st.download_button(
        label="Download processed image",
        data=file,
        file_name=name,
        mime="image/" + format
    )


def run():
    st.title("Image Segmentation by Emirhan")
    st.subheader("Upload your image and get an image with semantic segmentation")

    file = get_uploaded_file()
    if not file:
        return

    placeholder = st.empty()
    placeholder.info(
        "Processing your image..."
    )

    image = process_file(file)
    placeholder.empty()
    placeholder.image(image)

    filename = file.name
    format = re.findall("\..*$", filename)[0][1:]

    image = Image.fromarray(image)

    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_image = buf.getvalue()

    download_button(byte_image, filename, format)
    st.subheader('These are :blue[_labels_] :label:')
    placeholder = st.empty()
    labell = segment(
        Image.open(file),
        st.session_state["model"],
        st.session_state["feature_extractor"]
    )[1]
    a = []

    for i in labell:
        a.append(f"{label.trainId2label[i].name}")
    placeholder.info(f"{[aa for aa in a]}")


if __name__ == "__main__":
    init()
    run()
