import copy
import logging
import os
import os.path as osp
from functools import partial
from pydoc import locate
import shutil
import json
from traceback import print_exc
import uuid
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image

import whisper
import fire
import gradio as gr
import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, sizes
from gradio.components.image_editor import Brush

from cllm.agents.builtin import plans
from cllm.services.general.api import remote_logging
from cllm.agents import container, FILE_EXT
from cllm.utils import get_real_path, plain2md, md2plain
import openai

openai.api_base = os.environ.get("OPENAI_API_BASE", None)
openai.api_key = os.environ.get("OPENAI_API_KEY", None)


logging.basicConfig(
    filename="cllm.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

logger = logging.getLogger(__name__)

RESOURCE_ROOT = os.environ.get("CLIENT_ROOT", "./client_resources")


def is_image(file_path):
    ext = FILE_EXT["image"]
    _, extension = os.path.splitext(file_path)
    return extension[1:] in ext


def is_video(file_path):
    ext = FILE_EXT["video"]
    _, extension = os.path.splitext(file_path)
    return extension[1:] in ext


def is_audio(file_path):
    ext = FILE_EXT["audio"]
    _, extension = os.path.splitext(file_path)
    return extension[1:] in ext


def get_file_type(file_path):
    if is_image(file_path):
        if "mask" in file_path:
            return "mask"
        return "image"
    elif is_video(file_path):
        return "video"
    elif is_audio(file_path):
        return "audio"
    raise ValueError("Invalid file type")


def convert_dict_to_frame(data):
    import pandas

    outputs = []
    for k, v in data.items():
        output = {"Resource": k}
        if not isinstance(v, str):
            output["Type"] = str(v.__class__)
        else:
            output["Type"] = v
        outputs.append(output)
    if len(outputs) == 0:
        return None
    return pandas.DataFrame(outputs)


class Seafoam(ThemeBase.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.emerald,
        secondary_hue=colors.blue,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_sm,
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
        )
        super().set(
            body_background_fill_dark="#111111",
            button_primary_background_fill="*primary_300",
            button_primary_background_fill_hover="*primary_200",
            button_primary_text_color="black",
            button_secondary_background_fill="*secondary_300",
            button_secondary_background_fill_hover="*secondary_200",
            border_color_primary="#0BB9BF",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="10px",
        )


class InteractionLoop:
    def __init__(
        self,
        controller="cllm.agents.code.Controller",
    ):
        self.stream = True
        Controller = locate(controller)
        self.controller = Controller(stream=self.stream, interpretor_kwargs=dict())
        self.whisper = whisper.load_model("base")

    def _gen_new_name(self, r_type, ext="png"):
        this_new_uuid = str(uuid.uuid4())[:6]
        new_file_name = f"{this_new_uuid}_{r_type}.{ext}"
        return new_file_name

    def init_state(self):
        user_state = OrderedDict()
        user_state["resources"] = OrderedDict()
        user_state["history_msgs"] = []
        resources = OrderedDict()
        for item in sorted(os.listdir("./assets/resources")):
            if item.startswith("."):
                continue
            shutil.copy(
                osp.join("./assets/resources", item),
                osp.join(RESOURCE_ROOT, item),
            )
            resources[item] = get_file_type(item)
        # return user_state, user_state["resources"]
        return user_state, resources

    def add_file(self, user_state, history, file):
        if user_state.get("resources", None) is None:
            user_state["resources"] = OrderedDict()

        if file is None:
            return user_state, None, history, None
        # filename = os.path.basename(file.name)
        file = Path(file)
        ext = file.suffix[1:]
        if ext in FILE_EXT["image"]:
            ext = "png"
        r_type = get_file_type(file.name)
        new_filename = self._gen_new_name(r_type, ext)
        saved_path = get_real_path(new_filename)
        if ext in FILE_EXT["image"]:
            Image.open(file).convert("RGB").save(saved_path, "png")
            user_state["input_image"] = new_filename
        else:
            shutil.copy(file, saved_path)
        logger.info(f"add file: {saved_path}")
        user_state["resources"][new_filename] = r_type
        for key, val in user_state["resources"].items():
            if key == "prompt_points":
                user_state["resources"].pop(key)
                break
        history, _ = self.add_text(history, (saved_path,), role="human", append=False)
        history, _ = self.add_text(
            history, f"Recieved file {new_filename}", role="assistant", append=False
        )
        memory = convert_dict_to_frame(user_state["resources"])
        image_name = None
        if Path(saved_path).suffix[1:] in FILE_EXT["image"]:
            image_name = saved_path
        return user_state, image_name, history, memory

    def add_msg(self, history, text, audio, role="assistant", append=False):
        if text is not None and text.strip() != "":
            return self.add_text(history, text, role=role, append=append)
        elif audio is not None:
            return self.add_audio(history, audio, role=role, append=append)
        return history, ""

    def add_sketch(self, user_state, history, sketch):
        if user_state.get("resources", None) is None:
            user_state["resources"] = OrderedDict()

        if sketch is None or "layers" not in sketch:
            return user_state, None, history, None

        mask = None
        for layer in sketch["layers"]:
            alpha = layer[:, :, 3:] // 255
            if mask is None:
                mask = np.ones_like(layer[:, :, :3]) * 255
            mask = mask * (1 - alpha) + layer[:, :, :3] * alpha

        ext = "png"
        r_type = "scribble"
        new_filename = self._gen_new_name(r_type, ext)
        saved_path = get_real_path(new_filename)
        if ext in FILE_EXT["image"]:
            Image.fromarray(mask).save(saved_path, "png")
            user_state["sketch_image"] = new_filename

        logger.info(f"add file: {saved_path}")
        user_state["resources"][new_filename] = r_type
        history, _ = self.add_text(history, (saved_path,), role="human", append=False)
        history, _ = self.add_text(
            history, f"Recieved file {new_filename}", role="assistant", append=False
        )
        memory = convert_dict_to_frame(user_state["resources"])

        return user_state, history, memory

    def add_text(self, history, text, role="assistant", append=False):
        if history is None:
            return history, ""
        assert role in ["human", "assistant"]
        idx = 0
        if len(history) == 0 or role == "human":
            history.append([None, None])
        if role == "assistant":
            idx = 1
            if not append and history[-1][1] is not None:
                history.append([None, None])

        if append:
            history[-1][idx] = (
                text if history[-1][idx] is None else history[-1][idx] + text
            )
        else:
            history[-1][idx] = text
        if isinstance(text, str):
            logger.info(f"add text: {md2plain(text)}")

        return history, ""

    def add_audio(self, history, audio, role="assistant", append=False):
        assert role in ["human", "assistant"]
        result = self.whisper.transcribe(audio)
        text = result["text"]
        logger.info(f"add audio: {text}")
        return self.add_text(history, text, role=role, append=append)

    def plan(self, user_state, input_image, history, history_plan):
        logger.info(f"Task plan...")
        if user_state.get("resources", None) is None:
            user_state["resources"] = OrderedDict()

        request = history[-1][0]
        user_state["request"] = request
        if isinstance(request, str) and request.startswith("$"):
            solution = f'show$("{request[1:]}")'
        else:
            solution = self.controller.plan(request, state=user_state)
        print(f"request: {request}")
        if solution == self.controller.SHORTCUT:
            # md_text = "**Using builtin shortcut solution.**"
            history, _ = self.add_text(
                history, solution, role="assistant", append=False
            )
            user_state["solution"] = solution
            user_state["history_msgs"] = history
            yield user_state, input_image, history, [solution]
        elif isinstance(solution, str) and solution.startswith("show$"):
            user_state["solution"] = solution
            yield user_state, input_image, history, solution
        else:
            output_text = (
                "The whole process will take some time, please be patient.<br><br>"
            )
            history, _ = self.add_text(
                history, output_text, role="assistant", append=True
            )
            yield user_state, input_image, history, history_plan
            task_decomposition = next(solution)
            if task_decomposition in [None, [], ""]:
                output = "Error: unrecognized resource(s) in task decomposition."
                task_decomposition = "[]"
            else:
                output = task_decomposition

            output = f"**Task Decomposition:**\n{output}"
            output = plain2md(output)
            history, _ = self.add_text(history, output, role="assistant", append=True)
            user_state["task_decomposition"] = json.loads(task_decomposition)
            yield user_state, input_image, history, history_plan

            history, _ = self.add_text(
                history,
                plain2md("\n\n**Thoughs-on-Graph:**\n"),
                role="assistant",
                append=True,
            )
            yield user_state, input_image, history, history_plan
            solution_str = next(solution)
            logger.info(f"Thoughs-on-Graph: \n{solution_str}")
            if solution_str in [None, [], ""]:
                output = "Empty solution possibly due to some internal errors."
                solution_str = "[]"
            else:
                output = solution_str

            output_md = plain2md(output)
            history, _ = self.add_text(
                history, output_md, role="assistant", append=True
            )
            solution = json.loads(solution_str)
            user_state["solution"] = solution
            user_state["history_msgs"] = history
            yield user_state, input_image, history, solution

    def execute(self, user_state, input_image, history, history_plan):
        resources_state = user_state.get("resources", OrderedDict())
        solution = user_state.get("solution", None)
        if not solution:
            yield user_state, input_image, history, history_plan
            return
        logger.info(f"Tool execution...")
        if isinstance(solution, str) and solution.startswith("show$"):
            key = solution[7:-2]
            r_type = resources_state.get(key)
            if r_type is None:
                resource = f"{key} not found"
            resource = container.auto_type("None", r_type, key)
            history, _ = self.add_text(
                history, (resource.to_chatbot(),), role="assistant"
            )
            user_state["history_msgs"] = history
            yield user_state, input_image, history, history_plan
            return
        elif solution:
            results = self.controller.execute(solution, state=user_state)
            if not results:
                yield user_state, input_image, history, history_plan
                return

            user_state["outputs"] = []
            for result_per_step, executed_solutions, wrapped_outputs in results:
                tool_name = json.dumps(result_per_step[0], ensure_ascii=False)
                args = json.dumps(result_per_step[1], ensure_ascii=False)
                ret = json.dumps(result_per_step[2], ensure_ascii=False)
                history, _ = self.add_text(
                    history,
                    f"Call **{tool_name}:**<br>&nbsp;&nbsp;&nbsp;&nbsp;**Args**: {plain2md(args)}<br>&nbsp;&nbsp;&nbsp;&nbsp;**Ret**: {plain2md(ret)}",
                    role="assistant",
                )
                user_state["history_msgs"] = history
                user_state["executed_solutions"] = executed_solutions
                yield user_state, input_image, history, history_plan
                for _, output in enumerate(wrapped_outputs):
                    if output is None or output.value is None:
                        continue
                    if isinstance(output, container.File):
                        history, _ = self.add_text(
                            history,
                            f"Here is {output.filename}:",
                            role="assistant",
                        )
                        history, _ = self.add_text(
                            history, (output.to_chatbot(),), role="assistant"
                        )
                    user_state["outputs"].extend(wrapped_outputs)
                    user_state["history_msgs"] = history
                    yield user_state, input_image, history, history_plan

        else:
            yield user_state, input_image, history, history_plan

    def reply(self, user_state, history):
        logger.info(f"Make response...")
        executed_solution = user_state.get("executed_solutions", None)
        resources_state = user_state.get("resources", OrderedDict())
        solution = user_state.get("solution", None)
        memory = convert_dict_to_frame(resources_state)
        if isinstance(solution, str) and solution.startswith("show$"):
            return user_state, history, memory

        outputs = user_state.get("outputs", None)
        response, user_state = self.controller.reply(
            executed_solution, outputs, user_state
        )
        # prompt_mask_out = None
        for i, output in enumerate(response):
            if isinstance(output, container.File):
                history, _ = self.add_text(history, f"Here is [{output.filename}]: ")
                history, _ = self.add_text(history, (output.to_chatbot(),))
            elif i == 0:
                history, _ = self.add_text(history, output.to_chatbot())

        user_state["history_msgs"] = history
        return user_state, history, memory

    def vote(self, user_state, history, data: gr.LikeData):
        data_value = data.value
        if isinstance(data_value, dict):
            data_value = json.dumps(data_value)

        if data.liked:
            print("You upvoted this response: ", data_value)
            logger.info("You upvoted this response: " + data_value)
        else:
            print("You downvoted this response: ", data_value)
            logger.info("You downvoted this response: " + data_value)

        remote_logging(
            user_state.get("history_msgs", []),
            user_state.get("task_decomposition", ""),
            user_state.get("solution", []),
            data_value,
            data.liked,
        )

        msg = f"Thanks for your feedback! You feedback will contribute a lot to improving our ControlLLM."
        history, _ = self.add_text(history, msg)
        user_state["history_msgs"] = history
        return user_state, history

    def save_point(self, user_state, history, data: gr.SelectData):
        if isinstance(data, gr.LikeData):
            return self.vote(user_state, history, data)

        if not isinstance(data, gr.SelectData):
            return user_state, history

        resource_state = user_state.get("resources")
        input_image = user_state.get("input_image", None)
        if input_image is None:
            history, _ = self.add_text(history, "Please upload an image at first.")
            history, _ = self.add_text(history, plans.BUILTIN_SEG_BY_POINTS, "human")
            user_state["history_msg"] = history
            return user_state, history

        resource_state.pop(input_image, None)
        resource_state[input_image] = "image"

        history = history + [[plans.BUILTIN_SEG_BY_POINTS, None]]
        points = []
        if isinstance(points, str):
            points = json.loads(points)

        points.append(data.index)
        resource_state[json.dumps(points)] = "prompt_points"
        user_state["resources"] = resource_state
        return user_state, history


def on_switch_input(state_input, text, audio, disable=False):
    if state_input == "audio" or disable:
        return "text", gr.update(visible=True), gr.update(visible=False)
    return "audio", gr.update(visible=False), gr.update(visible=True)


def on_mask_submit(history):
    history = history + [(plans.BUILTIN_SEG_BY_MASK, None)]
    return history


def app(controller="cllm.agents.tog.Controller", https=False, **kwargs):
    loop = InteractionLoop(controller=controller)
    init_state, builtin_resources = loop.init_state()
    css = """
    code {
        font-size: var(--text-sm);
        white-space: pre-wrap;       /* Since CSS 2.1 */
        white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
        white-space: -pre-wrap;      /* Opera 4-6 */
        white-space: -o-pre-wrap;    /* Opera 7 */
        word-wrap: break-word;       /* Internet Explorer 5.5+ */
    }
    """
    with gr.Blocks(theme=Seafoam(), css=css) as demo:
        gr.HTML(
            """
            <div align='center'> <h1>ControlLLM </h1> </div>
            <p align="center"> A framework for multi-modal interaction which is able to control LLMs over invoking tools more accurately. </p>
            <p align="center"><a href="https://github.com/OpenGVLab/ControlLLM"><b>GitHub</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/abs/2311.11797"><b>ArXiv</b></a></p>
            """,
        )

        state_input = gr.State("text")
        user_state = gr.State(copy.deepcopy(init_state))
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(
                            [],
                            elem_id="chatbot",
                            avatar_images=[
                                "assets/human.png",
                                "assets/assistant.png",
                            ],
                            show_copy_button=True,
                            height=550,
                        )

                        with gr.Row():
                            with gr.Column(scale=12):
                                text = gr.Textbox(
                                    show_label=False,
                                    placeholder="Enter text and press enter, or upload an image.",
                                    container=False,
                                )
                                audio = gr.Audio(
                                    sources="microphone", type="filepath", visible=False
                                )
                            with gr.Column(scale=2, min_width=80):
                                submit = gr.Button("Submit", variant="primary")
                            with gr.Column(scale=1, min_width=40):
                                record = gr.Button("🎙️")
                            with gr.Column(scale=1, min_width=40):
                                upload_btn = gr.UploadButton(
                                    "📁",
                                    file_types=[
                                        "image",
                                        "video",
                                        "audio",
                                        ".pdf",
                                    ],
                                )

                        gr.Examples(
                            [
                                "Who are you?",
                                "How is about weather in Beijing",
                                "Describe the given image.",
                                "find the woman wearing the red skirt in the image",
                                "Generate a video that shows Pikachu surfing in waves.",
                                "How many horses are there in the image?",
                                "Can you erase the dog in the given image?",
                                "Remove the object based on the given mask.",
                                "Can you make a video of a serene lake with vibrant green grass and trees all around? And then create a webpage using HTML to showcase this video?",
                                "Generate an image that shows a beautiful landscape with a calm lake reflecting the blue sky and white clouds. Then generate a video to introduce this image.",
                                "replace the masked object with a cute yellow dog",
                                "Recognize the action in the video",
                                "Generate an image where a astronaut is riding a horse",
                                "Please generate a piece of music from the given image",
                                "Please give me an image that shows an astronaut riding a horse on mars.",
                                "What’s the weather situation in Berlin? Can you generate a new image that represents the weather in there?",
                                "Can you recognize the text from the image and tell me how much is Eggs Florentine?",
                                "Generate a piece of music for this video and dub this video with generated music",
                                "Generate a new image based on depth map from input image",
                                "Remove the cats from the image_1.png, image_2.png, image_3.png",
                                "I need the banana removed from the c4c40e_image.png, 9e867c_image.png, 9e13sc_image.png",
                                "I would be so happy if you could create a new image using the scribble from input image. The new image should be a tropical island with a dog. Write a detailed description of the given image. and highlight the dog in image",
                                "Please generate a piece of music and a new video from the input image",
                                "generate a new image conditioned on the segmentation from input image and the new image shows that a gorgeous lady is dancing",
                                "generate a new image with a different background but maintaining the same composition as input image",
                                "Generate a new image that shows an insect robot preparing a delicious meal. Then give me a video based on new image. Finally, dub the video with suitable background music.",
                                "Translate the text into speech: I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident that all men are created equal.I have a dream that one day on the red hills of Georgia the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood. I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice.",
                            ],
                            inputs=[text],
                        )
                        gr.Examples(
                            list(plans.BUILTIN_PLANS.keys()),
                            inputs=[text],
                            label="Builtin Examples",
                        )

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Click"):
                        click_image = gr.Image(
                            sources=["upload", "clipboard"],
                            interactive=True,
                            type="filepath",
                        )
                        with gr.Row():
                            click_image_submit_btn = gr.Button(
                                "Upload", variant="primary"
                            )
                        gr.Examples(
                            [
                                osp.join("./assets/resources", item)
                                for item in builtin_resources.keys()
                                if item.endswith(".png")
                            ],
                            inputs=[click_image],
                            label="File Examples",
                        )

                    with gr.Tab("Draw"):
                        sketch = gr.Sketchpad(
                            sources=(), brush=Brush(colors=["#000000"])
                        )
                        with gr.Row():
                            sketch_submit_btn = gr.Button("Upload", variant="primary")

                    with gr.Tab("Plan"):
                        planbot = gr.JSON(elem_classes="json")

                    with gr.Tab("Memory"):
                        memory_table = gr.DataFrame(
                            label="Memory",
                            headers=["Resource", "Type"],
                            row_count=5,
                            wrap=True,
                        )

        chatbot.like(
            loop.vote,
            [
                user_state,
                chatbot,
            ],
            [
                user_state,
                chatbot,
            ],
        )
        reply_inputs = [user_state, click_image, chatbot, planbot]
        reply_outputs = [user_state, chatbot, memory_table]

        add_text = [
            partial(loop.add_text, role="human"),
            [chatbot, text],
            [chatbot, text],
        ]

        text.submit(*add_text).then(loop.plan, reply_inputs, reply_inputs).then(
            loop.execute, reply_inputs, reply_inputs
        ).then(loop.reply, [user_state, chatbot], reply_outputs)

        add_msg = [
            partial(loop.add_msg, role="human"),
            [chatbot, text, audio],
            [chatbot, text],
        ]

        submit.click(*add_msg).then(
            partial(on_switch_input, disable=True),
            [state_input, text, audio],
            [state_input, text, audio],
        ).then(loop.plan, reply_inputs, reply_inputs).then(
            loop.execute, reply_inputs, reply_inputs
        ).then(
            loop.reply, [user_state, chatbot], reply_outputs
        )

        upload_btn.upload(
            loop.add_file,
            inputs=[user_state, chatbot, upload_btn],
            outputs=[user_state, click_image, chatbot, memory_table],
        )
        record.click(
            on_switch_input,
            [state_input, text, audio],
            [state_input, text, audio],
        )

        click_image.select(
            loop.save_point, [user_state, chatbot], [user_state, chatbot]
        ).then(loop.plan, reply_inputs, reply_inputs).then(
            loop.execute, reply_inputs, reply_inputs
        ).then(
            loop.reply, [user_state, chatbot], reply_outputs
        )

        click_image.upload(
            loop.add_file,
            inputs=[user_state, chatbot, click_image],
            outputs=[user_state, click_image, chatbot, memory_table],
        )
        click_image_submit_btn.click(
            loop.add_file,
            inputs=[user_state, chatbot, click_image],
            outputs=[user_state, click_image, chatbot, memory_table],
        )

        sketch_submit_btn.click(
            loop.add_sketch,
            inputs=[user_state, chatbot, sketch],
            outputs=[user_state, chatbot, memory_table],
        )

    if https:
        demo.queue().launch(
            server_name="0.0.0.0",
            ssl_certfile="./certificate/cert.pem",
            ssl_keyfile="./certificate/key.pem",
            ssl_verify=False,
            show_api=False,
            allowed_paths=[
                "assets/human.png",
                "assets/assistant.png",
            ],
            **kwargs,
        )
    else:
        demo.queue().launch(
            server_name="0.0.0.0",
            show_api=False,
            allowed_paths=[
                "assets/human.png",
                "assets/assistant.png",
            ],
            **kwargs,
        )


if __name__ == "__main__":
    os.makedirs(RESOURCE_ROOT, exist_ok=True)
    fire.Fire(app)
