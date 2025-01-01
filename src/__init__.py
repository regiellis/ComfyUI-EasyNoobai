import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# COMFYUI IMPORTS
import comfy.sd
import folder_paths
import torch

from .artists import ARTISTS
from .characters import CHARACTERS
from .e621_artists import E621_ARTISTS
from .e621_characters import E621_CHARACTERS

"""
EasyNoobai - Resources for implementation of EasyPony prompt sturcture
- https://civitai.com/articles/8962

"""


class EasyNoobai:

    GIRLS: List[str] = [f"{n} girl{'s' if n > 1 else ' solo'}" for n in range(1, 11)]
    BOYS: List[str] = [f"{n} boy{'s' if n > 1 else ' solo'}" for n in range(1, 11)]
    current_year = datetime.now().year
    YEARS = [f"{n} year{'' if n > 1 else ''}" for n in range(2000, current_year + 1)]

    NEG = " ".join(
        [
            "(((watermark))), worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing, displeasing"
        ]
    )
    NEG_EXTRA = ", ".join(
        [
            "ai-generated, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early",
            "very displeasing, displeasing, adversarial noise, what, off-topic, text, artist name, signature, username, logo",
            "watermark, copyright name, copyright symbol, low quality, lowres, jpeg artifacts, compression artifacts, blurry",
            "artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits, fewer digits, missing fingers",
            "censored, unfinished, bad proportions, bad perspective, monochrome, sketch, concept art, unclear, 2koma, 4koma,",
            "letterboxed, speech bubble, cropped",
        ]
    )
    NEG_BOOST = ", ".join(
        [
            "ai-generated, ai-assisted, stable diffusion, nai diffusion, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing",
            "displeasing, adversarial noise, unknown artist, banned artist, what, off-topic, artist request, text, artist name, signature, username, logo, watermark, copyright name, copyright symbol",
            "resized, downscaled, source larger, low quality, lowres, jpeg artifacts, compression artifacts, blurry, artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits",
            "fewer digits, missing fingers, censored, bar censor, mosaic censoring, missing, extra, fewer, bad, hyper, error, ugly, worst, tagme, unfinished, bad proportions, bad perspective, aliasing",
            "simple background, asymmetrical, monochrome, sketch, concept art, flat color, flat colors, simple shading, jaggy lines, traditional media \(artwork\), microsoft paint \(artwork\), ms paint \(medium\)",
            "unclear, photo, icon, multiple views, sequence, comic, 2koma, 4koma, multiple images, turnaround, collage, panel skew, letterboxed, framed, border, speech bubble, 3d, lossy-lossless, scan artifacts",
            "out of frame, cropped,",
        ]
    )

    NEG_ADDITIONAL = ", ".join([",(abstract:0.91), (doesnotexist:0.91)"])

    NEGATIVES: Dict[str, str] = {"Basic": NEG, "Extra": NEG_EXTRA, "Boost": NEG_BOOST}

    QUAILTY_BOOST = "masterpiece, best quality, good quality, very aesthetic, absurdres, newest, highres,"
    CINEMATIC = "(volumetric lighting:1.1, dof:1.1, depth of field:1.1)"

    CENSORSHIP = ", ".join(
        [
            "bar censor, censor, censor mosaic, censored, filter abuse",
            "heavily pixelated, instagram filter, mosaic censoring, over filter",
            "over saturated, over sharpened, overbrightened, overdarkened",
            "overexposed, overfiltered, oversaturated",
        ]
    )

    RESOLUTIONS: Dict[str, str] = {
        "9:16 - (768x1344)": "768x1344",
        "10:13 - (832x1216)": "832x1216",
        "4:5 - (896x1152)": "896x1152",
        "1:1 - (1024x1024)": "1024x1024",
        "2:3 - (1024x1536)": "1024x1536",
        "4:3 - (1152x896)": "1152x896",
        "3:2 - (1216x832)": "1216x832",
        "16:9 - (1344x768)": "1344x768",
        "3:2 - (1536x1024)": "1536x1024",
    }

    SHOT_TYPES: Dict[str, str] = {
        "Dutch Angle": "(dutch angle:1.15)",
        "From Above": "(from above:1.15)",
        "From Behind": "(from behind:1.15)",
        "From Below": "(from below:1.15)",
        "From Side": "(from side:1.15)",
        "Upside Down": "(upside down:1.15)",
        "High Up": "(high up:1.15)",
        "Multiple Views": "(multiple views:1.15)",
        "Sideways": "(sideways:1.15)",
        "Straight-On": "(straight-on:1.15)",
        "Three Quarter View": "(three quarter view:1.15)",
    }
    
    FRAMING: Dict[str, str] = {
        "Portrait": "portrait",
        "Upper Body": "upper body",
        "Lower Body": "lower body",
        "On Back": "on back, inverted",
        "Feet Out Of Frame": "feet out of frame",
        "Full Body": "full body",
        "Wide Shot": "wide shot",
        "Very Wide ": "very wide",
    }
    
    PERSPECTIVE: Dict[str, str] = {
        "Atmospheric Perspective": "atmospheric perspective",
        "Fisheye": "fisheye",
        "Panorama": "panorama",
        "Perspective": "perspective",
        "Vanishing Point": "vanishing point",
        "Variations": "variations",
    }
    
    FOCUS: Dict[str, str] = {
        "Animal Focus": "(animal focus:1.4)",
        "Armpit Focus": "(armpit focus:1.4)",
        "Ass Focus": "(ass focus:1.4)",
        "Back Focus": "(back focus:1.4)",
        "Book Focus": "(book focus:1.4)",
        "Breast Focus": "(breast focus:1.4)",
        "Cloud Focus": "(cloud focus:1.4)",
        "Eye Focus": "(eye focus:1.4)",
        "Food Focus": "(food focus:1.4)",
        "Foot Focus": "(foot focus:1.4)",
        "Hand Focus": "(hand focus:1.4)",
        "Hip Focus": "(hip focus:1.4)",
        "Male Focus": "(male focus:1.4)",
        "Monster Focus": "(monster focus:1.4)",
        "Navel Focus": "(navel focus:1.4)",
        "Object Focus": "(object focus:1.4)",
        "Other Focus": "(other focus:1.4)",
        "Plant Focus": "(plant focus:1.4)",
        "Pectoral Focus": "(pectoral focus:1.4)",
        "Solo Focus": "(solo focus:1.4)",
        "Vehicle Focus": "(vehicle focus:1.4)",
        "Text Focus": "(text focus:1.4)",
        "Thigh Focus": "(thigh focus:1.4)",
        "Weapon Focus": "(weapon focus:1.4)",
        "Wind Chime Focus": "(wind chime focus:1.4)",
    }

    WEIGHTS = {"LOW": ":0.6", "NORMAL": ":1.0", "HIGH": ":1.15", "VERY HIGH": ":1.4"}

    DEFAULT = " ".join(
        [
            "leaning against a wall in a alley, glitch hair, iridescent hair, holding gun, profile, backlighting, ",
            "scenery of a ruin city, patrol team in the background",
        ]
    )

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        PSONA_UI_EASY_NOOBAI: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Model": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "ONLY USE NOOBAI XL OR ILLUSTRIOUS-XL MODELS HERE."},
                ),
                "Stop at Clip Layer": (
                    "INT",
                    {"default": -2, "min": -2, "max": 10, "step": 1},
                ),
                "Resolution": (
                    list(EasyNoobai.RESOLUTIONS.keys()),
                    {
                        "default": "2:3 - (1024x1536)",
                        "tooltip": "Acts as a source filter.",
                    },
                ),
                "Batch Size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch.",
                    },
                ),
                "Character": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Danbooru Character List"},
                ),
                "Artist": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Danbooru Artist List"},
                ),
                "E621 Character": (
                    ["-"] + list(E621_CHARACTERS.keys()),
                    {"default": "-", "tooltip": "E621 Character List"},
                ),
                "E621 Artist": (
                    ["-"] + list(E621_ARTISTS.keys()),
                    {"default": "-", "tooltip": "E621 Artist List"},
                ),
                "Character Weight": (
                    ["-"] + list(EasyNoobai.WEIGHTS.keys()),
                    {"default": "-", "tooltip": "Weight of the Character"},
                ),
                "Artist Weight": (
                    ["-"] + list(EasyNoobai.WEIGHTS.keys()),
                    {"default": "-", "tooltip": "Weight of the Artist"},
                ),
                "Girl Characters": (
                    ["-"] + EasyNoobai.GIRLS,
                    {"default": "-", "tooltip": "Number of Girl Characters"},
                    {"default": "-", "tooltip": "Number of Girl Characters"},
                ),
                "Boy Characters": (
                    ["-"] + EasyNoobai.BOYS,
                    {"default": "-", "tooltip": "Number of Boy Characters"},
                ),
                "Mature Characters": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Changes Girl to Woman, Boy to Man"},
                ),
                "Year": (
                    ["-"] + EasyNoobai.YEARS,
                    {"default": "-", "tooltip": "Year of the character"},
                ),
                "Shot Type": (
                    ["-"] + list(EasyNoobai.SHOT_TYPES.keys()),
                    {"default": "-", "tooltip": "Type of shot"},
                ),
                "Framing": (
                    ["-"] + list(EasyNoobai.FRAMING.keys()),
                    {"default": "-", "tooltip": "Type of Chracter framing"},
                ),
                "Perspective": (
                    ["-"] + list(EasyNoobai.PERSPECTIVE.keys()),
                    {"default": "-", "tooltip": "Type of Chracter perspective"},
                ),
                "Focus": (
                    ["-"] + list(EasyNoobai.FOCUS.keys()),
                    {"default": "-", "tooltip": "Type of Chracter focus"},
                ),
                "Prompt": (
                    "STRING",
                    {"default": EasyNoobai.DEFAULT, "multiline": True},
                ),
                "Negative Prompt": (
                    ["-"] + list(EasyNoobai.NEGATIVES.keys()),
                    {
                        "default": "Basic",
                        "tooltip": "Select the type of negative prompt to use.",
                    },
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Will format the tag using \( and \)"},
                ),
                "Break Format": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Break the prompt into multiple tokens.",
                    },
                ),
                "SFW": (
                    "BOOLEAN",
                    {"default": False, "forceInput": False, "tooltip": "Safe for Work"},
                ),
                "Quality Boost (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Boost the quality of the image using negative prompts.",
                    },
                ),
                "Prefix QB (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Move the quality boost to the end of the prompt.",
                    },
                ),
                ("Cinematic (Beta)"): (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add cinematic elements to the prompt.",
                    },
                ),
            },
        }

        return PSONA_UI_EASY_NOOBAI

    RETURN_TYPES = (
        "MODEL",
        "VAE",
        "CLIP",
        "LATENT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "MODEL",
        "VAE",
        "CLIP",
        "LATENT",
        "PROMPT",
        "NEGATIVE",
    )
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
        False,
    )
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        prompt_elements, negative_elements = [], []
        ca_weight = (
            EasyNoobai.WEIGHTS[kwargs.get("Character Weight")]
            if kwargs.get("Character Weight") != "-"
            else ""
        )
        aa_weight = (
            EasyNoobai.WEIGHTS[kwargs.get("Artist Weight")]
            if kwargs.get("Artist Weight") != "-"
            else ""
        )

        resolution = self.parse_resolution(kwargs["Resolution"])
        kwargs["Boy Characters"] != "-" and prompt_elements.append(
            f"{kwargs['Boy Characters']},"
        )
        kwargs["Girl Characters"] != "-" and prompt_elements.append(
            f"{kwargs['Girl Characters']},"
        )

        # Add formatted tags to prompt elements
        if kwargs["Format Tag"]:
            kwargs.get("Character") != "-" and prompt_elements.append(
                f"\({self.format_tag(kwargs['Character'])}{ca_weight}\),"
            )
            kwargs.get("E621 Character") != "-" and prompt_elements.append(
                f"\({self.format_tag(kwargs['E621 Character'])}{ca_weight}\),"
            )
            kwargs.get("Artist") != "-" and prompt_elements.append(
                f"artist:{self.format_tag(kwargs['Artist'])}{aa_weight},"
            )
            kwargs.get("E621 Artist") != "-" and prompt_elements.append(
                f"artist:{self.format_tag(kwargs['E621 Artist'])}{aa_weight},"
            )
        else:
            kwargs.get("Character") != "-" and prompt_elements.append(
                f"{kwargs['Character']}{ca_weight},"
            )
            kwargs.get("E621 Character") != "-" and prompt_elements.append(
                f"{kwargs['E621 Character']}{ca_weight},"
            )
            kwargs.get("Artist") != "-" and prompt_elements.append(
                f"{self.format_tag(kwargs['Artist'])}{aa_weight},"
            )
            kwargs.get("E621 Artist") != "-" and prompt_elements.append(
                f"{self.format_tag(kwargs['E621 Artist'])}{aa_weight},"
            )

        kwargs["SFW"] and prompt_elements.append("(sfw:1.2),")
        kwargs["Year"] != "-" and prompt_elements.append(f"{kwargs['Year']},")
        
        shots = [EasyNoobai.SHOT_TYPES.get(kwargs.get("Shot Type", "-")), 
                 EasyNoobai.FRAMING.get(kwargs.get("Framing", "-")), 
                 EasyNoobai.PERSPECTIVE.get(kwargs.get("Perspective", "-")),
                 EasyNoobai.FOCUS.get(kwargs.get("Focus", "-"))]
        for shot in shots:
            if shot:
                prompt_elements.append(f"{shot},")



        kwargs.get("Prompt") and prompt_elements.append(kwargs["Prompt"] + ",")
        kwargs.get("suffix") and prompt_elements.append(kwargs["suffix"])

        # Construct negative elements

        if kwargs["Prefix QB (Beta)"]:
            kwargs["Quality Boost (Beta)"] and prompt_elements.insert(
                0,
                f"{self.QUAILTY_BOOST.strip()}{self.CINEMATIC.strip() if kwargs['Cinematic (Beta)'] else ''}",
            )
        else:
            kwargs["Quality Boost (Beta)"] and prompt_elements.append(
                f"{self.QUAILTY_BOOST.strip()}{self.CINEMATIC.strip() if kwargs['Cinematic (Beta)'] else ''}"
            )

        kwargs["SFW"] and negative_elements.append(self.CENSORSHIP.strip())

        final_prompt = " ".join(prompt_elements).lower()

        kwargs["Negative Prompt"] and negative_elements.append(
            self.NEGATIVES[kwargs["Negative Prompt"]]
        )

        if kwargs["Character"] != "-" or kwargs["E621 Character"] != "-":
            negative_elements.append(self.NEG_ADDITIONAL)

        if kwargs.get("Mature Characters", False):
            replacements = {
                "girls": "women",
                "girl": "woman",
                "boys": "men",
                "boy": "man",
            }
            for old, new in replacements.items():
                final_prompt = final_prompt.replace(old, new)

        clip = kwargs.get("Clip")
        last_clip_layer = kwargs.get("Stop at Clip Layer")
        latent = self.generate(resolution[0], resolution[1], kwargs["Batch Size"])[0]
        model = self.load_checkpoint(kwargs["Model"])
        checkpoint = model[0]
        clip = model[1]
        vae = model[2]

        modified_clip = self.modify_clip(clip, last_clip_layer)[0]
        final_negative = " ".join(negative_elements).strip()

        final_prompt = f"{prefix} {final_prompt} {suffix}"

        if kwargs["Break Format"]:
            final_prompt = self.format_prompt(final_prompt)
            final_negative = self.format_prompt(final_negative)

        return (
            checkpoint,
            vae,
            modified_clip,
            latent,
            final_prompt,
            final_negative,
        )

    @staticmethod
    def parse_resolution(resolution: str) -> tuple:
        dimensions = EasyNoobai.RESOLUTIONS[resolution].split("x")
        return int(dimensions[0]), int(dimensions[1])

    # add 'BREAK' keyword after every 70 characters making sure the last character is always a ","
    def format_prompt(self, prompt: str) -> str:
        prompt = prompt.replace(",", "")
        words = prompt.split()

        formatted_parts = []
        current_part = []
        current_length = 0

        for word in words:
            if current_length + len(word) > 70:
                formatted_parts.append(" ".join(current_part) + ", BREAK")
                current_part = [word]
                current_length = len(word)
            else:
                current_part.append(word)
                current_length += len(word) + 1  # +1 for the space

        if current_part:
            formatted_parts.append(" ".join(current_part))

        return " ".join(formatted_parts)

    def format_tag(self, tag: str) -> str:
        parts = tag.split("(")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))

        return "".join(formatted_parts)

    # FROM COMFYUI CORE
    def generate(self, width, height, batch_size=1) -> tuple:
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)

    # FROM COMFYUI CORE
    def modify_clip(self, clip, stop_at_clip_layer) -> tuple:
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)

    # FROM COMFYUI CORE
    def load_checkpoint(self, ckpt_name) -> tuple:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class NoobaiCharacters:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        PSONA_UI_NOOBAI_CHARACTERS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Character (Base)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character Base, Heavily influences the prompt",
                    },
                ),
                "Character (Secondary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character Secondary, influences the prompt",
                    },
                ),
                "Character (Tertiary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character Tertiary, influences the prompt less",
                    },
                ),
                "Character (Quaternary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character Quaternary, influences the prompt very little",
                    },
                ),
                "Character (Quinary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character Quinary, influences the prompt very little",
                    },
                ),
                "Weighed Average": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Weighed Average of all characters"},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Will format the tag using \( and \)"},
                ),
            },
        }

        return PSONA_UI_NOOBAI_CHARACTERS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CHARACTER PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        characters = [
            ("Character (Base)", 1.2),
            ("Character (Secondary)", 1.15),
            ("Character (Tertiary)", 1.125),
            ("Character (Quaternary)", 1.1),
            ("Character (Quinary)", 1),
        ]

        character_names = [char[0] for char in characters]
        if not any(kwargs.get(char) for char in character_names):
            return (prefix + suffix,)

        character_prompts = []
        selected_characters = []
        for char, weight in characters:
            char_value = kwargs.get(char)
            if char_value and char_value != "-":
                selected_characters.append((char_value, weight))

        avg_weight = self.calculate_avg_weight(selected_characters)
        use_avg_weight = kwargs.get("Weighed Average", False)

        if use_avg_weight:
            distributed_weights = self.distribute_weights(
                avg_weight, len(selected_characters)
            )
        else:
            distributed_weights = [weight for _, weight in selected_characters]

        for i, (char_value, _) in enumerate(selected_characters):
            weight_to_use = distributed_weights[i]
            formatted_char = f"{char_value}:{weight_to_use:.2f}"

            final_char = (
                f"\({formatted_char}\)"
                if kwargs.get("Format Tag")
                else f"{self.format_tag(formatted_char)}"
            )
            character_prompts.append(final_char)

        # Join character prompts with commas
        character_prompts_str = ", ".join(character_prompts)
        if suffix:
            character_prompts_str += f", {suffix}"

        if prefix:

            def find_insertion_point(text):
                commas = list(re.finditer(r",", text))
                if len(commas) >= 3:
                    return commas[1].end()
                return len(text)

            insert_index = find_insertion_point(prefix)
            final_prompt = f"{prefix[:insert_index].rstrip()} {character_prompts_str}, {prefix[insert_index:].lstrip()}"
        else:
            final_prompt = character_prompts_str

        final_prompt = re.sub(r"\s+", " ", final_prompt).strip()

        return (final_prompt,)

    @staticmethod
    def calculate_avg_weight(characters: List[Tuple[str, float]]) -> float:
        if not characters:
            return 0
        total_weight = sum(weight for _, weight in characters)
        return total_weight / len(characters)

    @staticmethod
    def distribute_weights(avg_weight: float, count: int) -> List[float]:
        if count == 1:
            return [min(round(avg_weight, 2), 1.15)]

        max_weight = 1.15
        min_weight = max(
            0, 2 * avg_weight - max_weight
        )  # Ensure min_weight is non-negative

        step = (max_weight - min_weight) / (count - 1)

        weights = [round(max_weight - i * step, 2) for i in range(count)]
        total = sum(weights)
        normalized_weights = [
            min(round(w * (avg_weight * count / total - 0.1), 2), 1.15) for w in weights
        ]

        return normalized_weights

    @staticmethod
    def format_tag(tag: str) -> str:
        parts = tag.split("(")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))

        return "".join(formatted_parts)


    @staticmethod
    def add_prefix(prompt_elements: List[str], prefix: str) -> str:
        prompt = " ".join(prompt_elements).lower()
        gender_words = ["girl", "girls", "boy", "boys", "woman", "women", "man", "men"]
        pattern = r"\b(" + "|".join(gender_words) + r")\b"

        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            index = match.end()
            return f"{prompt[:index]} {prefix} {prompt[index:]}"
        else:
            return f"{prefix} {prompt}"


class NoobaiArtists:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        PSONA_UI_NOOBAI_ARTISTS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Artist (Base)": (
                    ["-"] + list(ARTISTS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Artist Base, Heavily influences the prompt",
                    },
                ),
                "Artist (Secondary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Artist Secondary, influences the prompt",
                    },
                ),
                "Artist (Tertiary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Artist Tertiary, influences the prompt less",
                    },
                ),
                "Artist (Quaternary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Artist Quaternary, influences the prompt very little",
                    },
                ),
                "Artist (Quinary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Artist Quinary, influences the prompt very little",
                    },
                ),
                "Weighed Average": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Weighed Average of all artists"},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Will format the tag using \( and \)"},
                ),
            },
        }

        return PSONA_UI_NOOBAI_ARTISTS


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ARTIST PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        artists = [
            ("Artist (Base)", 1.2),
            ("Artist (Secondary)", 1.15),
            ("Artist (Tertiary)", 1.125),
            ("Artist (Quaternary)", 1.1),
            ("Artist (Quinary)", 1),
        ]

        artist_names = [artist[0] for artist in artists]
        if not any(kwargs.get(artist) for artist in artist_names):
            return (prefix + suffix,)

        artist_prompts = []
        selected_artists = []
        for artist, weight in artists:
            artist_value = kwargs.get(artist)
            if artist_value and artist_value != "-":
                selected_artists.append((artist_value, weight))

        avg_weight = self.calculate_avg_weight(selected_artists)
        use_avg_weight = kwargs.get("Weighed Average", False)

        if use_avg_weight:
            distributed_weights = self.distribute_weights(
                avg_weight, len(selected_artists)
            )
        else:
            distributed_weights = [weight for _, weight in selected_artists]

        for i, (artist_value, _) in enumerate(selected_artists):
            weight_to_use = distributed_weights[i]
            formatted_artist = f"{artist_value}:{weight_to_use:.2f}"

            final_artist = (
                f"(artist:{formatted_artist})"
                if kwargs.get("Format Tag")
                else f"{self.format_tag(formatted_artist)}"
            )
            artist_prompts.append(final_artist)

        # Join artist prompts with commas
        artist_prompts_str = ", ".join(artist_prompts)
        if suffix:
            artist_prompts_str += f", {suffix}"

        if prefix:
            character_names = list(CHARACTERS.keys())
            if prefix:
                insert_index = self.find_insertion_point(prefix, character_names)
                final_prompt = f"{prefix[:insert_index].rstrip()}, {artist_prompts_str}, {prefix[insert_index:].lstrip()}"
            else:
                final_prompt = artist_prompts_str    

        final_prompt = re.sub(r"\s+", " ", final_prompt).strip()

        return (final_prompt,)
    
    
    @staticmethod
    def find_insertion_point(text: str, character_names: List[str]) -> int:
        last_char_index: int = -1
        for name in character_names:
            pattern: str = rf'{re.escape(name)}(?:\s*\([^)]*\))?(?::[0-9.]+)?'
            matches: List[re.Match] = list(re.finditer(pattern, text))
            if matches:
                last_char_index = max(last_char_index, matches[-1].end())
        
        if last_char_index != -1:
            return last_char_index
        else:
            comma_match: Optional[re.Match] = re.search(r',\s*', text)
            if comma_match:
                return comma_match.end()
            else:
                return len(text)

    @staticmethod
    def calculate_avg_weight(characters: List[Tuple[str, float]]) -> float:
        if not characters:
            return 0
        total_weight = sum(weight for _, weight in characters)
        return total_weight / len(characters)


    @staticmethod
    def distribute_weights(avg_weight: float, count: int) -> List[float]:
        if count == 1:
            return [min(round(avg_weight, 2), 1.15)]

        max_weight = 1.15
        min_weight = max(
            0, 2 * avg_weight - max_weight
        )  # Ensure min_weight is non-negative

        step = (max_weight - min_weight) / (count - 1)

        weights = [round(max_weight - i * step, 2) for i in range(count)]
        total = sum(weights)
        normalized_weights = [
            min(round(w * (avg_weight * count / total - 0.1), 2), 1.15) for w in weights
        ]

        return normalized_weights


    @staticmethod
    def format_tag(tag: str) -> str:
        parts = tag.split("(")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))

        return "".join(formatted_parts)


class NoobaiE621Characters:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        E621_UI_CHARACTERS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with existing tags.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with additional tags.",
                    },
                ),
            },
            "required": {
                "Character (Base)": (
                    ["-"] + list(E621_CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Primary character"},
                ),
                "Character (Secondary)": (
                    ["-"] + list(E621_CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Secondary character"},
                ),
                "Character (Tertiary)": (
                    ["-"] + list(E621_CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Tertiary character"},
                ),
                "Lore Tags": (
                    "STRING",
                    {"default": "", "tooltip": "Additional lore tags"},
                ),
                "General Tags": (
                    "STRING",
                    {"default": "", "tooltip": "General tags (min 10)"},
                ),
            },
        }

        return E621_UI_CHARACTERS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("E621 CHARACTER TAGS",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        characters = [
            "Character (Base)",
            "Character (Secondary)",
            "Character (Tertiary)",
        ]

        character_tags = []
        for char in characters:
            char_value = kwargs.get(char)
            if char_value and char_value != "-":
                formatted_char = self.format_tag(char_value)
                character_tags.append(f"\(character:{formatted_char}\)")

        lore_tags = self.format_lore_tags(kwargs.get("Lore Tags", ""))
        general_tags = self.format_general_tags(kwargs.get("General Tags", ""))

        all_tags = character_tags + lore_tags + general_tags
        all_tags_str = ", ".join(all_tags)

        if prefix:
            final_prompt = f"{prefix} {all_tags_str}"
        else:
            final_prompt = all_tags_str

        if suffix:
            final_prompt += f" {suffix}"

        return (final_prompt.strip(),)

    @staticmethod
    def format_tag(tag: str) -> str:
        # Convert to lowercase, replace spaces with underscores, remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9_]", "", tag.lower().replace(" ", "_"))

    @staticmethod
    def format_lore_tags(lore_tags: str) -> List[str]:
        return [
            f"lore:{NoobaiE621Characters.format_tag(tag)}"
            for tag in lore_tags.split(",")
            if tag.strip()
        ]

    @staticmethod
    def format_general_tags(general_tags: str) -> List[str]:
        tags = [
            NoobaiE621Characters.format_tag(tag)
            for tag in general_tags.split(",")
            if tag.strip()
        ]
        # # Ensure at least 10 general tags
        # while len(tags) < 10:
        #     tags.append("")
        return tags


class NoobaiE621Artists:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        E621_UI_ARTISTS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with existing tags.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with additional tags.",
                    },
                ),
            },
            "required": {
                "Artist (Primary)": (
                    ["-"] + list(E621_ARTISTS.keys()),
                    {"default": "-", "tooltip": "Primary artist"},
                ),
                "Artist (Secondary)": (
                    ["-"] + list(E621_ARTISTS.keys()),
                    {"default": "-", "tooltip": "Secondary artist"},
                ),
                "Artist Aliases": (
                    "STRING",
                    {"default": "", "tooltip": "Additional artist aliases"},
                ),
                "Artist URLs": (
                    "STRING",
                    {"default": "", "tooltip": "Associated artist URLs"},
                ),
            },
        }

        return E621_UI_ARTISTS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("E621 ARTIST TAGS",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        artists = ["Artist (Primary)", "Artist (Secondary)"]

        artist_tags = []
        for artist in artists:
            artist_value = kwargs.get(artist)
            if artist_value and artist_value != "-":
                formatted_artist = self.format_tag(artist_value)
                artist_tags.append(f"artist:{formatted_artist}")

        alias_tags = self.format_aliases(kwargs.get("Artist Aliases", ""))
        url_tags = self.format_urls(kwargs.get("Artist URLs", ""))

        all_tags = artist_tags + alias_tags + url_tags
        all_tags_str = " ".join(all_tags)

        if prefix:
            final_prompt = f"{prefix} {all_tags_str}"
        else:
            final_prompt = all_tags_str

        if suffix:
            final_prompt += f" {suffix}"

        return (final_prompt.strip(),)

    @staticmethod
    def format_tag(tag: str) -> str:
        # Convert to lowercase, replace spaces with underscores, remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9_]", "", tag.lower().replace(" ", "_"))

    @staticmethod
    def format_aliases(aliases: str) -> List[str]:
        return [
            f"alias:{NoobaiE621Artists.format_tag(alias)}"
            for alias in aliases.split(",")
            if alias.strip()
        ]

    @staticmethod
    def format_urls(urls: str) -> List[str]:
        return [f"url:{url.strip()}" for url in urls.split(",") if url.strip()]
