import sys
from pathlib import Path
from typing import Dict
from enum import Enum


"""
Resources for implementation of prompt sturcture
  - https://civitai.com/articles/13283/pose-language-a-visual-guide-to-body-talk
  - https://civitai.com/articles/13746/pose-language-a-visual-guide-to-body-talk-p2
"""


class NoobaiPoses:

    class PoseTokens(Enum):
        CROUCHING = "crouching low, knees bent, heels lifted"
        SQUATTING = "squatting deep, knees bent, heels grounded"
        LOW_SQUAT = "low squat with hips near the ground"
        LEGS_APART = "legs apart, standing or sitting with legs open"
        WIDE_STANCE = "wide stance, feet apart in a strong posture"
        KNEES_UP = "knees up, one or both raised"
        ONE_KNEE_UP = "one knee up, seated or leaning"
        ONE_KNEE_ON_THE_GROUND = "one knee on the ground, other up, lunge-like"
        LEANING_FORWARD = "leaning forward, upper body tilted"
        HAND_ON_KNEE = "hand on knee, resting on one or both"
        HAND_BETWEEN_LEGS = "hand between legs, placed between thighs"
        SIDE_PROFILE_SQUAT = "side profile squat, viewed from the side"
        WALL_SQUAT = "wall squat, back against the wall"
        SITTING_ON_HEELS = "sitting on heels, hips resting on heels"
        PROVOCATIVE_POSE = "provocative pose, bold and attention-grabbing"
        SUGGESTIVE_POSE = "suggestive pose, subtle and hinting"
        SPREADING_LEGS = "spreading legs, deliberately apart"
        LEGS_UP = "legs up, lifted vertically or angled"
        ARCHED_BACK = "arched back, chest forward"
        LOOKING_BACK_SEDUCTIVELY = "looking back seductively, over-shoulder glance"
        RESTING_ON_ONE_KNEE = "resting on one knee, relaxed kneel"
        BENT_OVER = "bent over, torso forward, hips back"
        ALL_FOURS = "all fours, hands and knees on ground"
        LYING_ON_STOMACH = "lying on stomach, stretched out"
        LYING_ON_SIDE = "lying on side, one leg bent"
        THIGH_GAP = "thigh gap, thighs apart, not touching"
        LEGS_TOGETHER = "legs together, close and neat"
        HIPS_THRUST_FORWARD = "hips thrust forward or tilted ahead"
        STRADDLING = "straddling, legs open across an object or space"
        GRABBING_OWN_LEG = "grabbing own leg, reaching for thigh or calf"
        LEGS_CROSSED = "legs crossed, sitting or standing elegantly"
        SITTING_WITH_LEGS_SPREAD = "sitting with legs spread, relaxed or confident"
        ONE_LEG_RAISED = "one leg raised, poised or dynamic"
        HAND_ON_INNER_THIGH = "hand on inner thigh, subtle or intimate gesture"
        LIFTING_SKIRT = "lifting skirt, gently raising the hem"
        STANDING_POSE = "standing pose, upright and neutral"
        ONE_LEG_FORWARD = "one leg forward, stepping or leaning"
        LEG_OUTSTRETCHED = "leg outstretched, extended outward gracefully"
        BENT_KNEE = "bent knee, adding motion"
        KNEE_UP = "knee up, raised dynamically"
        HEEL_LIFT = "heel lift, raised slightly off the ground"
        WEIGHT_SHIFT = "weight shift, body angled to one side"
        LEG_UP_POSE = "leg up pose, raising a leg playfully or poised"
          
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        PSONA_UI_POSE_TOKEN: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Suffix to the prompt."},
                ),
            },
            "required": {
                "Descriptive (BETA)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Descriptive version of pose",
                    },
                ),
                "Pose Weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Weight of the pose in the prompt.",
                    },
                ),
                "Pose Set": (
                    ["-"] + list(NoobaiPoses.PoseTokens.__members__.keys()),
                    {
                        "default": "-",
                        "tooltip": "First set of Poses",
                    },
                ),
                "Format": (
                  "BOOLEAN",
                  {
                      "default":False,
                      "Tooltip": "Changes pose token into KEY, ex low squat > LOW_SQUAT"
                  }
                ),
                "Offset (Beta)": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "Word offset from the end of the prefix to inject the pose.",
                    },
                ),
            },
        }

        return PSONA_UI_POSE_TOKEN

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        pose_weight = kwargs.get("Pose Weight")
        pose_set_one = kwargs.get("Pose Set")
        if not kwargs.get("Descriptive (BETA)"):
            set_post = pose_set_one if kwargs.get('Format') else pose_set_one.lower().replace("_", " ")
            pose = set_post if set_post != "-" else ""
        else:
            pose = (
                ""
                if pose_set_one == "-"
                else NoobaiPoses.PoseTokens[pose_set_one].value
            )
        offset = kwargs.get("Offset", 0)

        # Construct the pose string
        pose_string = f" {pose}:{pose_weight}," if pose else ""

        # Inject the pose string into the prefix based on the offset
        if prefix and pose_string:
            prefix_words = prefix.split()
            insertion_index = max(0, len(prefix_words) - offset)
            prefix_words.insert(insertion_index, pose_string.strip())
            updated_prefix = " ".join(prefix_words)
        else:
            updated_prefix = prefix

        # Combine the updated prefix, suffix, and other elements
        final_prompt = " ".join(filter(None, [updated_prefix, suffix])).strip()

        return (final_prompt,)
