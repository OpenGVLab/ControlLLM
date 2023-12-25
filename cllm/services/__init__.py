from cllm.services.image_editing.api import (
    inpainting_ldm,
    inpainting_ldm_general,
    partial_image_editing,
    instruct_pix2pix,
    image_cropping,
    image_matting,
    draw_bbox_on_image,
)
from cllm.services.image_generation.api import (
    text2image,
    image2image,
    cannytext2image,
    linetext2image,
    hedtext2image,
    scribbletext2image,
    posetext2image,
    segtext2image,
    depthtext2image,
    normaltext2image,
)

from cllm.services.image_processing.api import (
    image2canny,
    image2line,
    image2hed,
    image2scribble,
    image2pose,
    image2depth,
    image2normal,
)
from cllm.services.image_perception.api import (
    object_detection,
    image_classification,
    ocr,
    segment_objects,
    visual_grounding,
    image_captioning,
    segment_by_mask,
    segment_by_points,
    set_image,
    segment_all,
    seg_by_mask,
    seg_by_points,
)
from cllm.services.video.api import (
    video_classification,
    video_captioning,
    image_audio_to_video,
    video_to_webpage,
    dub_video,
    image_to_video,
)
from cllm.services.audio.api import (
    text_to_music,
    text_to_speech,
    audio_classification,
)
from cllm.services.general.api import (
    select,
    count,
    remote_logging,
)
from cllm.services.nlp.api import (
    text_to_text_generation,
    title_generation,
    text_to_tags,
    question_answering_with_context,
    openai_chat_model,
    summarization,
    extract_location,
    sentiment_analysis,
    get_weather,
    summarize_weather_condition,
    get_time,
)
from cllm.services.vqa.api import image_qa

from fastapi import FastAPI
from .pool import ModelPool

app = FastAPI()
pool = ModelPool()
