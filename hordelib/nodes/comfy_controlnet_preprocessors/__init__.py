from . import canny, hed, midas, mlsd, openpose, uniformer, leres, mp_pose_hand, color, binary, pidinet, mp_face_mesh
from .util import HWC3, resize_image
import torch
import numpy as np
import cv2
import threading


def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)


def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    mask_list = [x.squeeze().numpy().astype(np.uint8) for x in torch.split(img_tensor, 1)]
    return mask_list
    # Thanks ChatGPT


def common_annotator_call(annotator_callback, tensor_image, *args):
    tensor_image_list = img_tensor_to_np(tensor_image)
    out_list = []
    out_info_list = []
    for tensor_image in tensor_image_list:
        call_result = annotator_callback(resize_image(HWC3(tensor_image)), *args)
        H, W, C = tensor_image.shape
        if type(annotator_callback) is openpose.OpenposeDetector:
            out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
            out_info_list.append(call_result[1])
        elif type(annotator_callback) is midas.MidasDetector:
            out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
            out_info_list.append(cv2.resize(HWC3(call_result[1]), (W, H), interpolation=cv2.INTER_AREA))
        else:
            out_list.append(cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA))
    if type(annotator_callback) is openpose.OpenposeDetector:
        return (out_list, out_info_list)
    elif type(annotator_callback) is midas.MidasDetector:
        return (out_list, out_info_list)
    else:
        return out_list


class Canny_Edge_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "l2gradient": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessors/edge_line"

    def detect_edge(self, image, low_threshold, high_threshold, l2gradient):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py
        np_detected_map = common_annotator_call(
            canny.CannyDetector(), image, low_threshold, high_threshold, l2gradient == "enable"
        )
        return (img_np_to_tensor(np_detected_map),)


class HED_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_boundary"

    CATEGORY = "preprocessors/edge_line"

    def detect_boundary(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hed2image.py
        np_detected_map = common_annotator_call(hed.HEDdetector(), image)
        return (img_np_to_tensor(np_detected_map),)


class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_scribble"

    CATEGORY = "preprocessors/edge_line"

    def transform_scribble(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_scribble2image.py
        np_img_list = img_tensor_to_np(image)
        out_list = []
        for np_img in np_img_list:
            np_detected_map = np.zeros_like(np_img, dtype=np.uint8)
            np_detected_map[np.min(np_img, axis=2) < 127] = 255
            out_list.append(np_detected_map)
        return (img_np_to_tensor(out_list),)


class Fake_Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_scribble"

    CATEGORY = "preprocessors/edge_line"

    def transform_scribble(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_fake_scribble2image.py
        np_detected_map_list = common_annotator_call(hed.HEDdetector(), image)
        out_list = []
        for np_detected_map in np_detected_map_list:
            np_detected_map = hed.nms(np_detected_map, 127, 3.0)
            np_detected_map = cv2.GaussianBlur(np_detected_map, (0, 0), 3.0)
            np_detected_map[np_detected_map > 4] = 255
            np_detected_map[np_detected_map < 255] = 0
            out_list.append(np_detected_map)
        return (img_np_to_tensor(out_list),)


class MIDAS_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
                "bg_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_depth(self, image, a, bg_threshold):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        depth_map_np, normal_map_np = common_annotator_call(midas.MidasDetector(), image, a, bg_threshold)
        return (img_np_to_tensor(depth_map_np),)


class MIDAS_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
                "bg_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_normal"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_normal(self, image, a, bg_threshold):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        depth_map_np, normal_map_np = common_annotator_call(midas.MidasDetector(), image, a, bg_threshold)
        return (img_np_to_tensor(normal_map_np),)


class LERES_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rm_nearest": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1, "step": 0.05}),
                "rm_background": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_depth(self, image, rm_nearest, rm_background):
        # Ref: https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L105
        depth_map_np = common_annotator_call(leres.apply_leres, image, rm_nearest, rm_background)
        return (img_np_to_tensor(depth_map_np),)


class MLSD_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Idk what should be the max value here since idk much about ML
                "score_threshold": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 2.0, "step": 0.05}),
                "dist_threshold": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessors/edge_line"

    def detect_edge(self, image, score_threshold, dist_threshold):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hough2image.py
        np_detected_map = common_annotator_call(mlsd.MLSDdetector(), image, score_threshold, dist_threshold)
        return (img_np_to_tensor(np_detected_map),)


class OpenPose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "detect_hand": (["disable", "enable"], {"default": "disable"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_pose"

    CATEGORY = "preprocessors/pose"

    def estimate_pose(self, image, detect_hand):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_pose2image.py
        np_detected_map, pose_info = common_annotator_call(openpose.OpenposeDetector(), image, detect_hand == "enable")
        return (img_np_to_tensor(np_detected_map),)


class Uniformer_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_seg2image.py
        np_detected_map = common_annotator_call(uniformer.UniformerDetector(), image)
        return (img_np_to_tensor(np_detected_map),)


class Media_Pipe_Hand_Pose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_pose": (["enable", "disable"], {"default": "enable"}),
                "detect_hands": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/pose"

    def detect(self, image, detect_pose, detect_hands):
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_pose2image.py
        np_detected_map = common_annotator_call(
            mp_pose_hand.apply_mediapipe, image, detect_pose == "enable", detect_hands == "enable"
        )
        return (img_np_to_tensor(np_detected_map),)


class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_faces": (
                    "INT",
                    {"default": 10, "min": 1, "max": 50, "step": 1},
                ),  # Which image has more than 50 detectable faces?
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/face_mesh"

    def detect(self, image, max_faces, min_confidence):
        np_detected_map = common_annotator_call(mp_face_mesh.generate_annotation, image, max_faces, min_confidence)
        return (img_np_to_tensor(np_detected_map),)


class Binary_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",), "threshold": ("INT", {"min": 0, "max": 255, "step": 1, "default": 0})}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_binary"

    CATEGORY = "preprocessors/edge_line"

    def transform_binary(self, image, threshold):
        np_detected_map = common_annotator_call(binary.apply_binary, image, threshold)
        return (img_np_to_tensor(np_detected_map),)


class Color_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_processed_pallete"

    CATEGORY = "preprocessors/color_style"

    def get_processed_pallete(self, image):
        np_detected_map = common_annotator_call(color.apply_color, image)
        return (img_np_to_tensor(np_detected_map),)


class PIDINET_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "preprocessors/edge_line"

    def detect_edge(self, image):
        np_detected_map = common_annotator_call(pidinet.apply_pidinet, image)
        return (img_np_to_tensor(np_detected_map),)


NODE_CLASS_MAPPINGS = {
    "CannyEdgePreprocessor": Canny_Edge_Preprocessor,
    "M-LSDPreprocessor": MLSD_Preprocessor,
    "HEDPreprocessor": HED_Preprocessor,
    "ScribblePreprocessor": Scribble_Preprocessor,
    "FakeScribblePreprocessor": Fake_Scribble_Preprocessor,
    "OpenposePreprocessor": OpenPose_Preprocessor,
    "MiDaS-DepthMapPreprocessor": MIDAS_Depth_Map_Preprocessor,
    "MiDaS-NormalMapPreprocessor": MIDAS_Normal_Map_Preprocessor,
    "LeReS-DepthMapPreprocessor": LERES_Depth_Map_Preprocessor,
    "SemSegPreprocessor": Uniformer_Preprocessor,
    "MediaPipe-HandPosePreprocessor": Media_Pipe_Hand_Pose_Preprocessor,
    "MediaPipe-FaceMeshPreprocessor": Media_Pipe_Face_Mesh_Preprocessor,
    "BinaryPreprocessor": Binary_Preprocessor,
    "ColorPreprocessor": Color_Preprocessor,
    "PiDiNetPreprocessor": PIDINET_Preprocessor,
}
