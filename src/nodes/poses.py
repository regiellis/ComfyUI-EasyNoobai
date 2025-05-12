import sys
import re
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
        
    CHAIN_INSERT_TOKEN = "[EN122112_CHAIN]"

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
                "Add Chain Insert": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "tooltip": f"If True, places '{NoobaiPoses.CHAIN_INSERT_TOKEN}' for the next chained node to insert its content."
                    }
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
                        "default": False,
                        "Tooltip": "Changes pose token into KEY, ex low squat > LOW_SQUAT",
                    },
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


    def _join_prompt_parts(self, *parts: str) -> str:
        """Helper to join non-empty prompt parts with ', ' and clean up."""
        # For this node, simple space joining might be more appropriate if it's just injecting one tag.
        # If it's meant to be a comma-separated tag, then the previous helper is fine.
        # Let's assume it should be a single tag that might be inserted.
        # If multiple poses were possible, then comma joining would be better.
        # For now, let's use a simpler join that doesn't add commas by default.
        filtered_parts = [p.strip() for p in parts if p and p.strip()]
        joined = " ".join(filtered_parts)
        # Clean up multiple spaces
        joined = re.sub(r'\s{2,}', ' ', joined).strip()
        return joined


    def construct(self, **kwargs) -> tuple:
        prefix_input_str = kwargs.get("prefix", "").strip()
        suffix_chain_input_str = kwargs.get("suffix", "").strip() # Suffix from this node's input
        add_insert_point_for_next_node = kwargs.get("Add Chain Insert Point", False)

        # --- 1. Generate THIS Node's CORE Content (the pose_string) ---
        pose_weight = kwargs.get("Pose Weight", 1.0)
        pose_set_one_name = kwargs.get("Pose Set") # This is the key/name of the pose
        
        pose_text = ""
        if pose_set_one_name != "-":
            if not kwargs.get("Descriptive (BETA)"):
                # Assuming pose_set_one_name is a string that might need formatting
                if kwargs.get("Format", True): # 'Format' here probably means 'replace underscores'
                    pose_text = pose_set_one_name # Already formatted if it's from a list of keys
                else:
                    pose_text = pose_set_one_name.lower().replace("_", " ")
            else:
                # Descriptive (BETA) logic
                try:
                    pose_text = NoobaiPoses.PoseTokens[pose_set_one_name].value
                except KeyError:
                    # Handle cases where pose_set_one_name might not be in PoseTokens
                    # For example, if "custom_raw_pose" is selected, maybe it implies using the string directly
                    if pose_set_one_name == "custom_raw_pose": # Or some other indicator for raw input
                        # If you have another input for raw pose text:
                        # pose_text = kwargs.get("Raw Pose Text", "").strip()
                        pose_text = "" # Or handle as error/default
                    else:
                        pose_text = "" # Default to empty if not found

        this_node_core_content_str = ""
        if pose_text:
            # Decide if the comma is part of the core content or added later
            # The original code had a comma: f" {pose_text}:{pose_weight},"
            # Let's keep it for now, assuming the pose is a distinct, weighted tag.
            this_node_core_content_str = f"{pose_text}:{pose_weight:.2f}" # Add comma later if needed by joining context
                                                                     # Removed leading space and trailing comma for now.

        # --- 2. Handle Prefix Input ---
        working_prompt_str = ""
        deferred_suffix_from_parent = ""
        offset_for_non_token_insert = kwargs.get("Offset", 0) # This node's specific offset

        if NoobaiPoses.CHAIN_INSERT_TOKEN in prefix_input_str:
            parts = prefix_input_str.split(NoobaiPoses.CHAIN_INSERT_TOKEN, 1)
            prefix_head = parts[0].strip()
            if len(parts) > 1:
                deferred_suffix_from_parent = parts[1].strip()
            
            # Insert this node's core content (pose_string)
            # If pose is a comma-separated tag, it should be joined with a comma.
            # If it's more like an adjective, space join.
            # Assuming it's a tag that should be comma-separated from other tags:
            temp_parts = [p for p in [prefix_head, this_node_core_content_str] if p and p.strip()]
            working_prompt_str = ", ".join(temp_parts).strip()
            working_prompt_str = re.sub(r'\s*,\s*', ', ', working_prompt_str) # Clean commas

        else:
            # TOKEN NOT FOUND: Use this node's original offset logic to inject pose_string
            if prefix_input_str and this_node_core_content_str:
                prefix_words = prefix_input_str.split()
                # Original logic: insert (len - offset) from end.
                # If offset = 0, insert at end. If offset = 1, insert before last word.
                # A positive offset means count from the end.
                # A negative offset could mean count from the start (adjust if that's the intent).
                # Let's assume positive offset = from end, 0 = at end.
                
                # Ensure offset is within bounds of prefix_words length
                num_prefix_words = len(prefix_words)
                
                # Convert offset to 0-based index from start for insertion
                # If offset = 0 (end), index = num_prefix_words
                # If offset = 1 (before last), index = num_prefix_words - 1
                # If offset = N (N from end), index = num_prefix_words - N
                # If offset is negative, e.g., -X, it means X from start. index = X (after validation)
                
                insertion_index = 0
                if offset_for_non_token_insert >= 0: # Counts from the end
                    insertion_index = num_prefix_words - offset_for_non_token_insert
                else: # Negative offset counts from the start
                    insertion_index = abs(offset_for_non_token_insert)
                
                insertion_index = max(0, min(num_prefix_words, insertion_index))

                prefix_words.insert(insertion_index, this_node_core_content_str) # Insert the pose tag
                working_prompt_str = " ".join(prefix_words).strip() # Join with spaces
                # After insertion, ensure it's a proper comma-separated list if that's the style.
                # This might be tricky if inserting mid-sentence. For now, assume it becomes part of a tag list.
                working_prompt_str = re.sub(r'\s*,\s*', ', ', working_prompt_str.replace(f" {this_node_core_content_str}", f", {this_node_core_content_str}")).strip()
                working_prompt_str = working_prompt_str.removeprefix(',').strip()


            elif this_node_core_content_str: # No prefix, just the pose
                working_prompt_str = this_node_core_content_str
            else: # No prefix, no pose
                working_prompt_str = prefix_input_str # (which is empty)
        
        working_prompt_str = working_prompt_str.strip()

        # --- 3. Add THIS node's CHAIN_INSERT_TOKEN (if toggled) ---
        # This node doesn't have complex "modifiers" like EasyNoobai. Token is added after its content.
        prompt_after_own_token_add = working_prompt_str
        if add_insert_point_for_next_node:
            if prompt_after_own_token_add: # If there's content, add token after a comma (if style is comma-sep)
                prompt_after_own_token_add = f"{prompt_after_own_token_add}, {NoobaiPoses.CHAIN_INSERT_TOKEN}"
            else: # If no content yet, token is the first thing
                prompt_after_own_token_add = NoobaiPoses.CHAIN_INSERT_TOKEN
        
        prompt_after_own_token_add = prompt_after_own_token_add.strip()
        
        # --- 4. Combine with deferred_suffix_from_parent and suffix_chain_input_str ---
        # Use the _join_prompt_parts helper, adapted for comma separation if that's the overall style
        temp_parts_for_final_join = [prompt_after_own_token_add, deferred_suffix_from_parent, suffix_chain_input_str]
        final_prompt_str = ", ".join([p for p in temp_parts_for_final_join if p and p.strip()]).strip()

        # --- 5. Final Cleanup ---
        final_prompt_str = re.sub(r'\s*,\s*', ', ', final_prompt_str).strip()
        final_prompt_str = re.sub(r',{2,}', ',', final_prompt_str)
        final_prompt_str = final_prompt_str.removeprefix(',').removesuffix(',').strip()
        # If CHAIN_INSERT_TOKEN is the only thing, or ends up with a comma before/after, clean it.
        final_prompt_str = final_prompt_str.replace(f", {NoobaiPoses.CHAIN_INSERT_TOKEN}", f" {NoobaiPoses.CHAIN_INSERT_TOKEN}")
        final_prompt_str = final_prompt_str.replace(f"{NoobaiPoses.CHAIN_INSERT_TOKEN},", f"{NoobaiPoses.CHAIN_INSERT_TOKEN} ")
        final_prompt_str = re.sub(r'\s{2,}', ' ', final_prompt_str).strip() # Consolidate spaces

        return (final_prompt_str if final_prompt_str else " ",)
