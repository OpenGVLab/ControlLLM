from cllm.agents.base import Action

BUILTIN_SEG_BY_POINTS = "Segment the given image based on the prompt points."
BUILTIN_SEG_BY_MASK = "Segment the given image based on the prompt mask."
# BUILTIN_REMOVE_BY_MASK = "Remove the object based on the given mask."
BUILTIN_IMAGE_TO_EDGE = "Generate the edge from the given image."
BUILTIN_GENERATE_SIMILAR_IMAGE = "Generate a new image similar to the input image"
# BUILTIN_GENERATE_SIMILAR_IMAGE2 = "Generate a similar image from the given image 2"
# BUILTIN_GENERATE_SIMILAR_IMAGE3 = "Image to image. 3"
BUILTIN_GENERATE_SIMILAR_IMAGE4 = "Generate a new image similar to image 4"
BUILTIN_GENERATE_IMAGE_HED = "Generate a new image based on HED result from input image"
BUILTIN_GENERATE_IMAGE_DEPTH = (
    "Generate a new image based on depth map from input image"
)
BUILTIN_GENERATE_IMAGE_OCR = "Please extract the text from the image"
BUILTIN_TEXT_EDGE_TO_IMAGE = "Generate an image based on the given edge map."
BUILTIN_GENERATE_IMAGE = "Generate a new image that shows a woman is skiing"
BUILTIN_IMAGE_TO_VIDEO = "Generate a video from the image"
BUILTIN_COUNT_OBJECTS = "Provide me with the count of bears in the input image"
BUILTIN_VIDEO_TO_WEBPAGE = "Generate a web page for input video"
BUILTIN_TEXT_TO_MUSIC = "Please generate a piece of music based on given prompt. Here is the prompt: An 80s driving pop song with heavy drums and synth pads in the background"
BUILTIN_IMAGE_ERASING1 = "Erase the wine glass from the photo"
BUILTIN_IMAGE_ERASING2 = "Erase the cats in the photo"
BUILTIN_IMAGE_CROPPING = "Crop the cats from the photo"
BUILTIN_IMAGE_SEG = "give me the mask of elephant."
BUILTIN_IMAGE_HIGHLIGHT = "highlight the elephant."
BUILTIN_TEXT_SPEECH = "translate text into speech"
BUILTIN_DUBBING = "dub this video with the given audio"
BUILTIN_COUNT_OBJECTS2 = "Count the horse in the image."
BUILTIN_IMAGE_TO_VIDEO2 = "Generate an image that shows a serene and beautiful landscape with a calm lake reflecting the blue sky and white clouds. Then generate a video to introduce this image."
BUILTIN_IMAGE_TO_VIDEO3 = "Create a visual and auditory representation of a peaceful and scenic landscape. The image should depict a serene and beautiful landscape with a calm lake reflecting the blue sky. The music should match the image. Finally, combine the image and the music into a video that showcases the beauty of nature."
BUILTIN_VIDEO_CLS = "Recognize the action in the video"
BUILTIN_VIDEO_CLS = "Recognize the action in the video"
BUILTIN_AUDIO_CLS = "Recognize the event in this audio"
BUILTIN_IMAGE2MUSIC = "Generate a piece of music for this image"
BUILTIN_VIDEO2MUSIC = (
    "Generate a piece of music for this video and dub the video with generated music"
)
BUILTIN_SPEECH_TO_TEXT = "transcribe the speech into text"

BUILTIN_PLANS = {
    # BUILTIN_REMOVE_BY_MASK: [
    #     [
    #         Action(
    #             tool_name="image_inpainting",
    #             inputs={"image": "image", "mask": "image.mask"},
    #             outputs=["<GENERATED>-0"],
    #         )
    #     ]
    # ],
    BUILTIN_IMAGE_TO_EDGE: [
        [
            Action(
                tool_name="image_to_edge",
                inputs={"image": "image"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_TEXT_EDGE_TO_IMAGE: [
        [
            Action(
                tool_name="image_captioning",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-prompt"],
            ),
            Action(
                tool_name="edge_text_to_image",
                inputs={
                    "edge": "image.edge",
                    "text": "<TOOL-GENERATED>-prompt",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_GENERATE_SIMILAR_IMAGE: [
        [
            Action(
                tool_name="image_to_edge",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-edge"],
            ),
            Action(
                tool_name="image_captioning",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-prompt"],
            ),
            Action(
                tool_name="edge_text_to_image",
                inputs={
                    "edge": "<TOOL-GENERATED>-edge",
                    "text": "<TOOL-GENERATED>-prompt",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    # BUILTIN_GENERATE_SIMILAR_IMAGE2: [
    #     [
    #         Action(
    #             tool_name="image_captioning",
    #             inputs={"image": "image"},
    #             outputs=["<TOOL-GENERATED>-prompt"],
    #         ),
    #         Action(
    #             tool_name="text_to_image",
    #             inputs={"text": "<TOOL-GENERATED>-prompt"},
    #             outputs=["<GENERATED>-0"],
    #         ),
    #     ]
    # ],
    # BUILTIN_GENERATE_SIMILAR_IMAGE3: [
    #     [
    #         Action(
    #             tool_name="image_to_image",
    #             inputs={"image": "image"},
    #             outputs=["<GENERATED>-0"],
    #         ),
    #     ]
    # ],
    BUILTIN_GENERATE_IMAGE_HED: [
        [
            Action(
                tool_name="image_to_hed",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-image_to_hed-hed-0"],
            ),
            Action(
                tool_name="hed_text_to_image",
                inputs={
                    "text": "beautiful mountains and sunset",
                    "hed": "<TOOL-GENERATED>-image_to_hed-hed-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_GENERATE_IMAGE_DEPTH: [
        [
            Action(
                tool_name="image_captioning",
                inputs={
                    "image": "image",
                },
                outputs=["<TOOL-GENERATED>-image_captioning-text-0"],
            ),
            Action(
                tool_name="image_to_depth",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-image_to_depth-depth-0"],
            ),
            Action(
                tool_name="depth_text_to_image",
                inputs={
                    "text": "<TOOL-GENERATED>-image_captioning-text-0",
                    "depth": "<TOOL-GENERATED>-image_to_depth-depth-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_GENERATE_IMAGE_OCR: [
        [
            Action(
                tool_name="optical_character_recognition",
                inputs={"image": "image"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_COUNT_OBJECTS: [
        [
            Action(
                tool_name="object_detection",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-object_detection-bbox-0"],
            ),
            Action(
                tool_name="select_bbox",
                inputs={
                    "bbox_list": "<TOOL-GENERATED>-object_detection-bbox-0",
                    "condition": "bear",
                },
                outputs=["<TOOL-GENERATED>-select_bbox-bbox-0"],
            ),
            Action(
                tool_name="count_objects",
                inputs={"bbox_list": "<TOOL-GENERATED>-select_bbox-bbox-0"},
                outputs=["<GENERATED>-0"],
            ),
        ],
        [
            Action(
                tool_name="image_question_answering",
                inputs={
                    "text": "Provide me with the count of bears in the input image",
                    "image": "image",
                },
                outputs=["<GENERATED>-1"],
            )
        ],
    ],
    BUILTIN_VIDEO_TO_WEBPAGE: [
        [
            Action(
                tool_name="video_captioning",
                inputs={"video": "video"},
                outputs=["<TOOL-GENERATED>-text-0"],
            ),
            Action(
                tool_name="text_to_music",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<TOOL-GENERATED>-text_to_music-audio-0"],
            ),
            Action(
                tool_name="dub_video",
                inputs={
                    "video": "video",
                    "audio": "<TOOL-GENERATED>-text_to_music-audio-0",
                },
                outputs=["<TOOL-GENERATED>-dub_video-video-0"],
            ),
            Action(
                tool_name="title_generation",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<TOOL-GENERATED>-text-1"],
            ),
            Action(
                tool_name="text_to_tags",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<TOOL-GENERATED>-tags-0"],
            ),
            Action(
                tool_name="video_to_webpage",
                inputs={
                    "video": "<TOOL-GENERATED>-dub_video-video-0",
                    "title": "<TOOL-GENERATED>-text-1",
                    "tags": "<TOOL-GENERATED>-tags-0",
                    "description": "<TOOL-GENERATED>-text-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_TEXT_TO_MUSIC: [
        [
            Action(
                tool_name="text_to_music",
                inputs={
                    "text": "An 80s driving pop song with heavy drums and synth pads in the background"
                },
                outputs=["<GENERATED>-audio-0"],
            )
        ]
    ],
    BUILTIN_IMAGE_ERASING1: [
        [
            Action(
                tool_name="image_instance_segmentation",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-image_instance_segmentation-mask-0"],
            ),
            Action(
                tool_name="select_mask",
                inputs={
                    "mask_list": "<TOOL-GENERATED>-image_instance_segmentation-mask-0",
                    "condition": "wine glass",
                },
                outputs=["<TOOL-GENERATED>-select_mask-mask-1"],
            ),
            Action(
                tool_name="image_inpainting",
                inputs={
                    "image": "image",
                    "mask": "<TOOL-GENERATED>-select_mask-mask-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_IMAGE_ERASING2: [
        [
            Action(
                tool_name="image_instance_segmentation",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-image_instance_segmentation-mask-0"],
            ),
            Action(
                tool_name="select_mask",
                inputs={
                    "mask_list": "<TOOL-GENERATED>-image_instance_segmentation-mask-0",
                    "condition": "cat",
                },
                outputs=["<TOOL-GENERATED>-select_mask-mask-0"],
            ),
            Action(
                tool_name="image_inpainting",
                inputs={
                    "image": "image",
                    "mask": "<TOOL-GENERATED>-select_mask-mask-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_IMAGE_CROPPING: [
        [
            Action(
                tool_name="object_detection",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-object_detection-bbox-0"],
            ),
            Action(
                tool_name="select_bbox",
                inputs={
                    "bbox_list": "<TOOL-GENERATED>-object_detection-bbox-0",
                    "condition": "cat",
                },
                outputs=["<TOOL-GENERATED>-select_bbox-bbox-0"],
            ),
            Action(
                tool_name="image_cropping",
                inputs={
                    "image": "image",
                    "object": "<TOOL-GENERATED>-select_bbox-bbox-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_IMAGE_SEG: [
        [
            Action(
                tool_name="image_instance_segmentation",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-image_instance_segmentation-mask-0"],
            ),
            Action(
                tool_name="select_mask",
                inputs={
                    "mask_list": "<TOOL-GENERATED>-image_instance_segmentation-mask-0",
                    "condition": "elephant",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_IMAGE_HIGHLIGHT: [
        [
            Action(
                tool_name="object_detection",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-object_detection-bbox-0"],
            ),
            Action(
                tool_name="select_bbox",
                inputs={
                    "bbox_list": "<TOOL-GENERATED>-object_detection-bbox-0",
                    "condition": "elephant",
                },
                outputs=["<TOOL-GENERATED>-select_bbox-bbox-0"],
            ),
            Action(
                tool_name="highlight_object_on_image",
                inputs={
                    "image": "image",
                    "bbox": "<TOOL-GENERATED>-select_bbox-bbox-0",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_TEXT_SPEECH: [
        [
            Action(
                tool_name="text_to_speech",
                inputs={
                    "text": "Hope is the thing with feathers That perches in the soul, And sings the tune without the words, And never stops at all"
                },
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_DUBBING: [
        [
            Action(
                tool_name="dub_video",
                inputs={"video": "video", "audio": "audio"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_GENERATE_SIMILAR_IMAGE4: [
        [
            Action(
                tool_name="segment_anything",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-seg"],
            ),
            Action(
                tool_name="image_captioning",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-prompt"],
            ),
            Action(
                tool_name="segmentation_text_to_image",
                inputs={
                    "segmentation": "<TOOL-GENERATED>-seg",
                    "text": "<TOOL-GENERATED>-prompt",
                },
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_GENERATE_IMAGE: [
        [
            Action(
                tool_name="text_to_image",
                inputs={"text": "a woman is skiing"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_IMAGE_TO_VIDEO: [
        [
            Action(
                tool_name="image_to_video",
                inputs={"image": "image"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_COUNT_OBJECTS2: [
        [
            Action(
                tool_name="object_detection",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-object_detection-bbox-0"],
            ),
            Action(
                tool_name="select_bbox",
                inputs={
                    "bbox_list": "<TOOL-GENERATED>-object_detection-bbox-0",
                    "condition": "horse",
                },
                outputs=["<TOOL-GENERATED>-select_bbox-bbox-0"],
            ),
            Action(
                tool_name="count_objects",
                inputs={"bbox_list": "<TOOL-GENERATED>-select_bbox-bbox-0"},
                outputs=["<GENERATED>-0"],
            ),
        ],
        [
            Action(
                tool_name="image_question_answering",
                inputs={
                    "text": "Provide me with the count of horses in the input image",
                    "image": "image",
                },
                outputs=["<GENERATED>-1"],
            )
        ],
    ],
    BUILTIN_IMAGE_TO_VIDEO2: [
        [
            Action(
                tool_name="text_to_image",
                inputs={
                    "text": "A serene and beautiful landscape with a calm lake reflecting the blue sky and white clouds."
                },
                outputs=["<GENERATED>-0"],
            ),
        ],
        [
            Action(
                tool_name="image_captioning",
                inputs={"image": "<GENERATED>-0"},
                outputs=["<TOOL-GENERATED>-text-0"],
            ),
            Action(
                tool_name="text_to_speech",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<TOOL-GENERATED>-text_to_speech-audio-0"],
            ),
            Action(
                tool_name="image_audio_to_video",
                inputs={
                    "image": "<GENERATED>-0",
                    "audio": "<TOOL-GENERATED>-text_to_speech-audio-0",
                },
                outputs=["<GENERATED>-1"],
            ),
        ],
    ],
    BUILTIN_IMAGE_TO_VIDEO3: [
        [
            Action(
                tool_name="text_to_image",
                inputs={
                    "text": "A serene and beautiful landscape with a calm lake reflecting the blue sky."
                },
                outputs=["<GENERATED>-0"],
            ),
        ],
        [
            Action(
                tool_name="image_captioning",
                inputs={"image": "<GENERATED>-0"},
                outputs=["<TOOL-GENERATED>-text-0"],
            ),
            Action(
                tool_name="text_to_music",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<GENERATED>-1"],
            ),
        ],
        [
            Action(
                tool_name="image_to_video",
                inputs={
                    "image": "<GENERATED>-0",
                },
                outputs=["<TOOL-GENERATED>-image_to_video-video-0"],
            ),
            Action(
                tool_name="dub_video",
                inputs={
                    "video": "<TOOL-GENERATED>-image_to_video-video-0",
                    "audio": "<GENERATED>-1",
                },
                outputs=["<GENERATED>-2"],
            ),
        ],
    ],
    BUILTIN_VIDEO_CLS: [
        [
            Action(
                tool_name="video_classification",
                inputs={"video": "video"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_AUDIO_CLS: [
        [
            Action(
                tool_name="audio_classification",
                inputs={"audio": "audio"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    BUILTIN_IMAGE2MUSIC: [
        [
            Action(
                tool_name="image_captioning",
                inputs={"image": "image"},
                outputs=["<TOOL-GENERATED>-text-0"],
            ),
            Action(
                tool_name="text_to_music",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<GENERATED>-0"],
            ),
        ]
    ],
    BUILTIN_VIDEO2MUSIC: [
        [
            Action(
                tool_name="video_captioning",
                inputs={"video": "video"},
                outputs=["<TOOL-GENERATED>-text-0"],
            ),
            Action(
                tool_name="text_to_music",
                inputs={"text": "<TOOL-GENERATED>-text-0"},
                outputs=["<GENERATED>-0"],
            ),
        ],
        [
            Action(
                tool_name="dub_video",
                inputs={
                    "video": "video",
                    "audio": "<GENERATED>-0",
                },
                outputs=["<GENERATED>-1"],
            ),
        ],
    ],
    BUILTIN_SEG_BY_POINTS: [
        [
            Action(
                tool_name="image_segmentation_by_points",
                inputs={"image": "image", "prompt_points": "prompt_points"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
    # BUILTIN_SEG_BY_MASK: [
    #     [
    #         Action(
    #             tool_name='image_segmentation_by_mask',
    #             inputs={'image': 'image', 'prompt_mask': 'prompt_mask'},
    #             outputs=['<GENERATED>-0'],
    #         )
    #     ]
    # ],
    BUILTIN_SPEECH_TO_TEXT: [
        [
            Action(
                tool_name="speech_to_text",
                inputs={"speech": "audio"},
                outputs=["<GENERATED>-0"],
            )
        ]
    ],
}


def load_builtin_plans(path):
    import json

    plans = json.load(open(path, "r"))
    processed_plan = {}
    for query, actions in plans.items():
        actions2 = []
        for ac in actions[0]:
            actions2.append(
                Action(
                    tool_name=ac["tool_name"],
                    inputs=ac["inputs"],
                    outputs=ac["outputs"],
                ),
            )
        processed_plan[query] = [actions2]
    return processed_plan
