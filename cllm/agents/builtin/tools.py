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
    text_to_video,
)
from cllm.services.audio.api import (
    text_to_music,
    text_to_speech,
    audio_classification,
    speech_to_text,
)

# from cllm.services.sam.api import (
#     segment_by_mask,
#     segment_by_points,
#     set_image,
#     segment_all,
# )
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
from cllm.agents.base import Tool, DataType


QUESTION_ANSWERING_TOOLS = [
    Tool(
        name="image_question_answering",
        description="answers a question about an image",
        domain=Tool.Domain.VISUAL_QUESTION_ANSWERING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image containing the information",
            ),
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the question about the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="response",
                type=DataType.TEXT,
                description="output response",
            )
        ],
        model=image_qa,
    ),
    Tool(
        name="get_weather",
        description="Query the weather conditions by given location. For example: what is the weather in Beijing? how cold is in New York? etc.",
        domain=Tool.Domain.QUESTION_ANSWERING,
        args=[
            Tool.Argument(
                name="location",
                type=DataType.LOCATION,
                description="the location where the weather is to be queried",
            ),
        ],
        returns=[
            Tool.Argument(
                name="result",
                # type=DataType.WEATHER,
                type=DataType.WEATHER,
                description="weather information",
            )
        ],
        model=get_weather,
    ),
    Tool(
        name="get_time",
        description="get current date",
        domain=Tool.Domain.QUESTION_ANSWERING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="input text",
            ),
        ],
        returns=[
            Tool.Argument(
                name="response",
                type=DataType.TIME,
                description="output time",
            )
        ],
        model=get_time,
    ),
    # Tool(
    #     name="calculator",
    #     description="It can solve mathematics problems and support various mathematical expressions: from basic arithmetic to more complex expressions.",
    #     domain=Tool.Domain.QUESTION_ANSWERING,
    #     args=[
    #         Tool.Argument(
    #             name="text",
    #             type=DataType.TEXT,
    #             description="input instructions",
    #         ),
    #     ],
    #     returns=[
    #         Tool.Argument(
    #             name="result",
    #             type=DataType.TEXT,
    #             description="result about weather",
    #         )
    #     ],
    #     model=None,
    # ),
]

IMAGE_CAPTIONING_TOOLS = [
    Tool(
        name="image_captioning",
        description='Generate a caption or description for the image. It can generate a detailed description that can be used for image perception and image generation. For example: a) you can use this tool when you want to know what is it in the image"; and b) when you want to generate a new image similar or resemble to input.png, you can use `image_captioning` to obtain the description about image input.png.',
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be captioned",
            ),
        ],
        returns=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the description for the input image",
            )
        ],
        model=image_captioning,
    ),
]

IMAGE_EDITING_TOOLS = [
    Tool(
        name="partial_image_editing",
        description="Given the mask denoting the region to edit,  Edit the given image at local region. Useful when you want to replace an object via a mask image. "
        "like: replace the masked object with a dog. ",
        domain=Tool.Domain.IMAGE_EDITING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be edited",
            ),
            Tool.Argument(
                name="mask",
                type=DataType.MASK,
                description="the mask image representing the editing position",
            ),
            Tool.Argument(
                name="prompt",
                type=DataType.TEXT,
                description="the prompt specified the edition",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the edited image",
            )
        ],
        model=partial_image_editing,
    ),
    Tool(
        name="text_image_editing",
        description="Edit the given image based on the text prompt.",
        domain=Tool.Domain.IMAGE_EDITING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be edited",
            ),
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the prompt specified the edition",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the edited image",
            )
        ],
        model=instruct_pix2pix,
    ),
    Tool(
        name="image_inpainting",
        description="inpaint the region of the image based on the given mask. For example: remove the dog in the image, erase the spoon in given image, etc.",
        domain=Tool.Domain.IMAGE_EDITING,
        usages=["remove some objects"],
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be inpainted",
            ),
            Tool.Argument(
                name="mask",
                type=DataType.MASK,
                description="the segmentation mask for the inpainting region",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the processed image",
            )
        ],
        model=inpainting_ldm_general,
    ),
    Tool(
        name="highlight_object_on_image",
        description="This tool is usually used after `object_detection` `visual_grounding` and `select_bbox`. Useful when you want to: 1) highlight the region of interest on the image; 2) know where the object is. For example: highlight the elephant from image, locate the dog in the image, find the spoon in given image, detect if the object is present in the image, etc.",
        domain=Tool.Domain.IMAGE_EDITING,
        usages=["highlight the region of interest on the image"],
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
            Tool.Argument(
                name="bbox",
                type=DataType.BBOX,
                description="the bounding boxes that need to be drawn on the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="result",
                type=DataType.IMAGE,
                description="the new image on which the tool highlight the the region of interest by bounding boxes",
            )
        ],
        model=draw_bbox_on_image,
    ),
    Tool(
        name="image_cropping",
        description="Crop the image based on the given bounding box. Useful when you want to crop the dog in the image, crop the spoon in given image, etc.",
        domain=Tool.Domain.IMAGE_EDITING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
            Tool.Argument(
                name="object",
                type=DataType.BBOX,
                description="the detected object",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the cropped image",
            )
        ],
        model=image_cropping,
    ),
    # Tool(
    #     name="mask_image",
    #     description="Mask the background from the image based on the given mask. For example: mask anything except the dog in the image, extract the spoon from given image without any inpainting, etc.",
    #     domain=Tool.Domain.IMAGE_EDITING,
    #     args=[
    #         Tool.Argument(
    #             name="image",
    #             type=DataType.IMAGE,
    #             description="the image to be processed",
    #         ),
    #         Tool.Argument(
    #             name="mask",
    #             type=DataType.MASK,
    #             description="the mask of the matted region",
    #         ),
    #     ],
    #     returns=[
    #         Tool.Argument(
    #             name="image",
    #             type=DataType.IMAGE,
    #             description="the matted image",
    #         )
    #     ],
    #     model=image_matting,
    # ),
]

IMAGE_GENERATION_TOOLS = [
    Tool(
        name="text_to_image",
        description="generate an image based on the given description.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=text2image,
    ),
    Tool(
        name="image_to_image",
        description="generate an new image based on the given image.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the given image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=image2image,
    ),
    Tool(
        name="line_text_to_image",
        description="generate an image based on the given description and line map.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="line",
                type=DataType.LINE,
                description="the line map outlining the line of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=linetext2image,
    ),
    Tool(
        name="hed_text_to_image",
        description="generate an image based on the given description and HED map (holistically-nested edge detection).",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="hed",
                type=DataType.HED,
                description="the HED map outlining the edge of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=hedtext2image,
    ),
    Tool(
        name="scribble_text_to_image",
        description="generate an image based on the given description and the scribble.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="scribble",
                type=DataType.SCRIBBLE,
                description="the scribble outlining the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=scribbletext2image,
    ),
    Tool(
        name="pose_text_to_image",
        description="generate an image based on the given description and the pose.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="pose",
                type=DataType.POSE,
                description="the pose of the human in the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=posetext2image,
    ),
    Tool(
        name="segmentation_text_to_image",
        description="generate an image based on the given description and segmentation mask.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="segmentation",
                type=DataType.SEGMENTATION,
                description="the segmentation mask describing the structure of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=segtext2image,
    ),
    Tool(
        name="edge_text_to_image",
        description="generate an image based on the given description and edge map.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="edge",
                type=DataType.EDGE,
                description="the edge map describing the structure of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=cannytext2image,
    ),
    Tool(
        name="depth_text_to_image",
        description="generate an image based on the given description and depth map.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="depth",
                type=DataType.DEPTH,
                description="the depth map describing the structure of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=depthtext2image,
    ),
    Tool(
        name="normal_text_to_image",
        description="generate an image based on the given description and normal map.",
        domain=Tool.Domain.IMAGE_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the text describing the image",
            ),
            Tool.Argument(
                name="normal",
                type=DataType.NORMAL,
                description="the normal map describing the structure of the image",
            ),
        ],
        returns=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the generated image",
            )
        ],
        model=normaltext2image,
    ),
]

IMAGE_TRANSFORM_TOOLS = [
    Tool(
        name="image_to_edge",
        description="get the edge map of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="edge",
                type=DataType.EDGE,
                description="the edge map of the image",
            )
        ],
        model=image2canny,
    ),
    Tool(
        name="image_to_line",
        description="get the line map of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="line",
                type=DataType.LINE,
                description="the line map of the image",
            )
        ],
        model=image2line,
    ),
    Tool(
        name="image_to_hed",
        description="get the HED map of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="hed",
                type=DataType.HED,
                description="the hed map of the image",
            )
        ],
        model=image2hed,
    ),
    Tool(
        name="image_to_scribble",
        description="get the scribble of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="scribble",
                type=DataType.SCRIBBLE,
                description="the scribble of the image",
            )
        ],
        model=image2scribble,
    ),
    Tool(
        name="image_to_pose",
        description="Get the pose of the image. It is usually used in image generation conditioned on pose map from input image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="pose",
                type=DataType.POSE,
                description="the pose of the image",
            )
        ],
        model=image2pose,
    ),
    Tool(
        name="image_to_depth",
        description="get the depth map of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="depth",
                type=DataType.DEPTH,
                description="the depth map",
            )
        ],
        model=image2depth,
    ),
    Tool(
        name="image_to_normal",
        description="get the normal map of the image.",
        domain=Tool.Domain.IMAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="normal",
                type=DataType.NORMAL,
                description="the normal map",
            )
        ],
        model=image2normal,
    ),
]

IMAGE_PERCEPTION_TOOLS = [
    Tool(
        name="object_detection",
        description="detect all the objects in the image.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that contains the objects",
            ),
        ],
        returns=[
            Tool.Argument(
                name="object",
                type=DataType.BBOX,
                description="the detected objects in json format. "
                "example output: [\{'score': 0.9994931221008301, 'label': 'dog', 'box': \{'xmin': 466, 'ymin': 301, 'xmax': 1045, 'ymax': 583\}\}]",
            )
        ],
        model=object_detection,
    ),
    Tool(
        name="image_classification",
        description="classify the objects in the image.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        usages=["ask about the class of the image"],
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that contains the objects",
            ),
        ],
        returns=[
            Tool.Argument(
                name="category",
                type=DataType.CATEGORY,
                description="the categories in json format. "
                "example output: [\{'score': 0.9, 'label': 'dog'\}]",
            )
        ],
        model=image_classification,
    ),
    Tool(
        name="video_classification",
        description="Classify the video and detect the actions in the video.",
        domain=Tool.Domain.VIDEO_PERCEPTION,
        usages=["ask about the class of the video"],
        args=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="the given video",
            ),
        ],
        returns=[
            Tool.Argument(
                name="category",
                type=DataType.CATEGORY,
                description="the categories in json format. "
                "example output: [\{'score': 0.9, 'label': 'Playing basketball'\}]",
            )
        ],
        model=video_classification,
    ),
    Tool(
        name="image_instance_segmentation",
        description="segment the common objects in the given image.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be segmented",
            ),
        ],
        returns=[
            Tool.Argument(
                name="mask", type=DataType.MASK, description="the output mask"
            )
        ],
        model=segment_objects,
    ),
    Tool(
        name="image_segmentation_by_mask",
        description="segment the given image with the prompt mask.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be segmented",
            ),
            Tool.Argument(
                name="prompt_mask",
                type=DataType.MASK,
                description="the prompt mask that guides the segmentation",
            ),
        ],
        returns=[
            Tool.Argument(
                name="mask", type=DataType.MASK, description="the output mask"
            )
        ],
        model=seg_by_mask,
    ),
    Tool(
        name="image_segmentation_by_points",
        description="segment the given image with the prompt points.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be segmented",
            ),
            Tool.Argument(
                name="prompt_points",
                type=DataType.POINT,
                description="the prompt points that guides the segmentation",
            ),
        ],
        returns=[
            Tool.Argument(
                name="mask", type=DataType.MASK, description="the output mask"
            )
        ],
        model=seg_by_points,
    ),
    Tool(
        name="segment_anything",
        description="Segment the given image without other inputs. This tool return the segmentation map for input image. The segmentation can be used to generate a new image.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be segmented",
            ),
        ],
        returns=[
            Tool.Argument(
                name="segmentation",
                type=DataType.SEGMENTATION,
                description="the output segmentation",
            )
        ],
        model=segment_all,
    ),
    Tool(
        name="visual_grounding",
        description="Visual Grounding (VG) aims to locate the most relevant object or region in an image, based on a natural language query. The query can be a phrase, a sentence or even a multi-round dialogue.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be processed",
            ),
            Tool.Argument(
                name="query",
                type=DataType.TEXT,
                description="a query that can be a phrase, a sentence",
            ),
        ],
        returns=[
            Tool.Argument(
                name="bbox",
                type=DataType.BBOX,
                description="the detected bounding boxes for ",
            )
        ],
        model=visual_grounding,
    ),
    Tool(
        name="optical_character_recognition",
        description="Optical Character Recognition (OCR) is the process that converts an image of text into a machine-readable text format.",
        domain=Tool.Domain.IMAGE_PERCEPTION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="the image that need to be processed",
            )
        ],
        returns=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="the recognized text",
            )
        ],
        model=ocr,
    ),
]

GENERAL_TOOLS = [
    Tool(
        name="select_category",
        description="select the target classes in category list with the given condition.",
        domain=Tool.Domain.GENERAL,
        usages=["pick out the objects with the same type"],
        args=[
            Tool.Argument(
                name="category_list",
                type=DataType.CATEGORY,
                description="the list to be processed",
            ),
            Tool.Argument(
                name="condition",
                type=DataType.TEXT,
                description="the condition to select objects",
            ),
        ],
        returns=[
            Tool.Argument(
                name="target_category_result",
                type=DataType.CATEGORY,
                description="the selected list",
            )
        ],
        model=select,
    ),
    Tool(
        name="select_bbox",
        description="select the bounding boxes with the given condition.",
        domain=Tool.Domain.GENERAL,
        usages=["filter out the bounding boxes with the same type"],
        args=[
            Tool.Argument(
                name="bbox_list",
                type=DataType.BBOX,
                description="the bounding box list to be processed",
            ),
            Tool.Argument(
                name="condition",
                type=DataType.TEXT,
                description="the condition to select objects",
            ),
        ],
        returns=[
            Tool.Argument(
                name="result",
                type=DataType.BBOX,
                description="the selected bbox list",
            )
        ],
        model=select,
    ),
    Tool(
        name="select_mask",
        description="select the masks with the given condition.",
        domain=Tool.Domain.GENERAL,
        args=[
            Tool.Argument(
                name="mask_list",
                type=DataType.MASK,
                description="the list to be processed",
            ),
            Tool.Argument(
                name="condition",
                type=DataType.TEXT,
                description="the condition to select objects",
            ),
        ],
        returns=[
            Tool.Argument(
                name="result",
                type=DataType.MASK,
                description="the selected mask list",
            )
        ],
        model=select,
    ),
    Tool(
        name="count_categories",
        description="count target categories in the given list.",
        domain=Tool.Domain.GENERAL,
        args=[
            Tool.Argument(
                name="category_list",
                type=DataType.CATEGORY,
                description="the list to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="length",
                type=DataType.TEXT,
                description="the length of the given list, return in the string format."
                "Example: The length of the given list is 10",
            )
        ],
        model=count,
    ),
    Tool(
        name="count_objects",
        description="count target objects in the given list. It is useful when you want to count the number of objects in the image",
        domain=Tool.Domain.GENERAL,
        args=[
            Tool.Argument(
                name="bbox_list",
                type=DataType.BBOX,
                description="the bounding box list to be counted",
            ),
        ],
        returns=[
            Tool.Argument(
                name="length",
                type=DataType.TEXT,
                description="the length of the given list, return in the string format."
                "Example: The length of the given list is 10",
            )
        ],
        model=count,
    ),
    Tool(
        name="count_masks",
        description="count target mask in the given list.",
        domain=Tool.Domain.GENERAL,
        args=[
            Tool.Argument(
                name="mask_list",
                type=DataType.MASK,
                description="the list to be processed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="length",
                type=DataType.TEXT,
                description="the length of the given list, return in the string format."
                "Example: The length of the given list is 10",
            )
        ],
        model=count,
    ),
]

VIDEO_TOOLS = [
    # VIDEO
    Tool(
        name="video_captioning",
        description='Generate a caption or description for video. It can generate a detailed description that can be used for video perception and video generation. For example: a) you can use this tool when you want to know what happened in the video"; and b) when you want to generate tags for input video, you can use translate description obtained from `image_captioning` into tags.',
        domain=Tool.Domain.VIDEO_PERCEPTION,
        args=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="the video to be captioned.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="caption",
                type=DataType.TEXT,
                description="the caption or description of input video.",
            )
        ],
        model=video_captioning,
    ),
    Tool(
        name="image_audio_to_video",
        description="Generate a video with speech to introduce the image.",
        domain=Tool.Domain.VIDEO_GENERATION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="The input image to be introduced.",
            ),
            Tool.Argument(
                name="audio",
                type=DataType.AUDIO,
                description="The audio contained the speech of image description.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="Generated video that can introduce the image with speech",
            )
        ],
        model=image_audio_to_video,
    ),
    Tool(
        name="image_to_video",
        description="Generate a video based on image.",
        domain=Tool.Domain.VIDEO_GENERATION,
        args=[
            Tool.Argument(
                name="image",
                type=DataType.IMAGE,
                description="The input image.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="Generated video from the input image.",
            )
        ],
        model=image_to_video,
    ),
    Tool(
        name="video_to_webpage",
        description="Generate a web page to promote and introduce the video.",
        domain=Tool.Domain.VIDEO_PROCESSING,
        args=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="The input image to be introduced.",
            ),
            Tool.Argument(
                name="title",
                type=DataType.TITLE,
                description="The title of video.",
            ),
            Tool.Argument(
                name="tags",
                type=DataType.TAGS,
                description="The tags of video.",
            ),
            Tool.Argument(
                name="description",
                type=DataType.TEXT,
                description="The description of video.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="html_code",
                type=DataType.HTML,
                description="Generated HTML webpage with code that can introduce the video with speech.",
            )
        ],
        model=video_to_webpage,
    ),
    Tool(
        name="dub_video",
        description="Dub the input video with given audio track.",
        domain=Tool.Domain.VIDEO_EDITING,
        args=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="The input image to be introduced.",
            ),
            Tool.Argument(
                name="audio",
                type=DataType.AUDIO,
                description="The audio of video.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="Output video with designated audio.",
            )
        ],
        model=dub_video,
    ),
    Tool(
        name="text_to_video",
        description="It takes as input a natural language description and produces a video matching that description",
        domain=Tool.Domain.VIDEO_GENERATION,
        args=[
            Tool.Argument(
                name="prompt",
                type=DataType.TEXT,
                description="the text describing the image",
            )
        ],
        returns=[
            Tool.Argument(
                name="video",
                type=DataType.VIDEO,
                description="the generated video",
            )
        ],
        model=text_to_video,
    ),
]

AUDIO_TOOLS = [
    # AUDIO
    Tool(
        name="text_to_music",
        description="Generate music condioned on input text/prompt. For example, you can use this tool when you want to generate music for a poem, generate a piece of music from image.",
        domain=Tool.Domain.AUDIO_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="Input text for music generation.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="music",
                type=DataType.AUDIO,
                description="Generated music conditioned on text.",
            )
        ],
        model=text_to_music,
    ),
    Tool(
        name="text_to_speech",
        description="Create natural-sounding speech from text, where the speech can be generated in multiple languages and for multiple speakers",
        domain=Tool.Domain.AUDIO_GENERATION,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text that will be translated into speech.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="speech",
                type=DataType.AUDIO,
                description="Generated speech or voice conditioned on text.",
            )
        ],
        model=text_to_speech,
    ),
    Tool(
        name="audio_classification",
        description="Audio classification is the task of assigning a label or class to a given audio. It can be used for recognizing which command a user is giving or the emotion of a statement, as well as identifying a speaker.",
        domain=Tool.Domain.AUDIO_PERCEPTION,
        args=[
            Tool.Argument(
                name="audio",
                type=DataType.AUDIO,
                description="The input audio that will be classified.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="speech",
                type=DataType.CATEGORY,
                description="The recognized categories in json format.",
            )
        ],
        model=audio_classification,
    ),
    Tool(
        name="speech_to_text",
        description="Transcribe the speech to text. It can be used for speech recognition, speech to text, speech to subtitle, etc.",
        domain=Tool.Domain.AUDIO_PERCEPTION,
        args=[
            Tool.Argument(
                name="speech",
                type=DataType.AUDIO,
                description="The input audio that will be classified.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The transcribed text.",
            )
        ],
        model=speech_to_text,
    ),
]

NLP_TOOLS = [
    # Text
    Tool(
        name="text_to_text_generation",
        description="Text to text generation. It can be used for sentence acceptability judgment, Sentiment analysis, Paraphrasing/sentence similarity, Natural language inference, Sentence completion, Word sense disambiguation, Question answering.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text",
            ),
        ],
        returns=[
            Tool.Argument(
                name="answer",
                type=DataType.TEXT,
                description="Generated answer for given input.",
            )
        ],
        model=text_to_text_generation,
    ),
    Tool(
        name="title_generation",
        description="Generate a title for given text.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text",
            ),
        ],
        returns=[
            Tool.Argument(
                name="title",
                type=DataType.TITLE,
                description="Generated title based given sentences.",
            )
        ],
        model=title_generation,
    ),
    Tool(
        name="openai_chat_model",
        description="Answer the question by Large Language Model. It is useful for tasks such as generating content, answering questions, engaging in conversations and providing explanations. However, it still has some limitations. For example, it can not directly access the up-to-date information like time, weather, etc.",
        domain=Tool.Domain.QUESTION_ANSWERING,
        args=[
            Tool.Argument(
                name="input_msg",
                type=DataType.TEXT,
                description="The input text",
            )
        ],
        returns=[
            Tool.Argument(
                name="answer",
                type=DataType.TEXT,
                description="Generated answer based given text.",
            )
        ],
        model=openai_chat_model,
    ),
    Tool(
        name="summarization",
        description="Summarize sentences, long narratives, articles, papers, textbooks.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text to be Summarized.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="summarized_text",
                type=DataType.TEXT,
                description="Summarized text.",
            )
        ],
        model=summarization,
    ),
    Tool(
        name="text_to_tags",
        description="Predict the tags of text, article and papers by using the their textual content as input",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text to be Summarized.",
            ),
        ],
        returns=[
            Tool.Argument(
                name="tags",
                type=DataType.TAGS,
                description="The extracted tags from input text",
            )
        ],
        model=text_to_tags,
    ),
    Tool(
        name="named_entity_recognition",
        description="Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text from which the named entities are extracted",
            ),
        ],
        returns=[
            Tool.Argument(
                name="tags",
                type=DataType.TAGS,
                description="The extracted entities",
            )
        ],
        model=None,
    ),
    Tool(
        name="sentiment_analysis",
        description="Sentiment analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text to be analyzed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The sentiment of text",
            )
        ],
        model=sentiment_analysis,
    ),
    Tool(
        name="extract_location",
        description="Extracts the locale name from the text. For example, if the text is 'what is the weather in Beijing', the tool will return 'Beijing'. If the text is 'Samuel ppops in a happy plce called Berlin which happens to be Kazakhstan', the tool will return 'Berlin,Kazakhstan'.",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="text",
                type=DataType.TEXT,
                description="The input text to be analyzed",
            ),
        ],
        returns=[
            Tool.Argument(
                name="location",
                type=DataType.LOCATION,
                description="The extracted location from text",
            )
        ],
        model=extract_location,
    ),
    Tool(
        name="summarize_weather_condition",
        description="Translate the json formatted weather information into the text that human can understand. For example, when you want to generate a new image based on weather information",
        domain=Tool.Domain.NATURAL_LANGUAGE_PROCESSING,
        args=[
            Tool.Argument(
                name="weather",
                type=DataType.WEATHER,
                description="weather condition",
            )
        ],
        returns=[
            Tool.Argument(
                name="weather_summary",
                type=DataType.TEXT,
                description="the weather summary",
            )
        ],
        model=summarize_weather_condition,
    ),
]

TOOLS = (
    QUESTION_ANSWERING_TOOLS
    + IMAGE_CAPTIONING_TOOLS
    + IMAGE_EDITING_TOOLS
    + IMAGE_GENERATION_TOOLS
    + IMAGE_TRANSFORM_TOOLS
    + IMAGE_PERCEPTION_TOOLS
    + GENERAL_TOOLS
    + VIDEO_TOOLS
    + AUDIO_TOOLS
    + NLP_TOOLS
)
TOOLS = {tool.name: tool for tool in TOOLS}

if __name__ == "__main__":
    tools = []
    for tool in TOOLS.values():
        tools.append(tool.dict())
    import json

    with open("tools.json", "w") as f:
        json.dump(tools, f, indent=4)
