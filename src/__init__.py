import sys
import re
from pathlib import Path
from typing import Dict, List, Union
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
    
    GIRLS: List[str] = [
        f"{n} girl{'s' if n > 1 else ' solo'}" for n in range(1, 11)
    ]
    BOYS: List[str] = [
        f"{n} boy{'s' if n > 1 else ' solo'}" for n in range(1, 11)
    ]
    current_year = datetime.now().year
    YEARS = [f"{n} year{'' if n > 1 else ''}" for n in range(2000, current_year + 1)]
    

    NEG = " ".join(["worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing, displeasing"])
    NEG_EXTRA = ", ".join(["ai-generated, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early",
                          "very displeasing, displeasing, adversarial noise, what, off-topic, text, artist name, signature, username, logo",
                          "watermark, copyright name, copyright symbol, low quality, lowres, jpeg artifacts, compression artifacts, blurry",
                          "artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits, fewer digits, missing fingers",
                          "censored, unfinished, bad proportions, bad perspective, monochrome, sketch, concept art, unclear, 2koma, 4koma,",
                          "letterboxed, speech bubble, cropped"
    ])
    NEG_BOOST = ", ".join([
          "ai-generated, ai-assisted, stable diffusion, nai diffusion, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing",
          "displeasing, adversarial noise, unknown artist, banned artist, what, off-topic, artist request, text, artist name, signature, username, logo, watermark, copyright name, copyright symbol",
          "resized, downscaled, source larger, low quality, lowres, jpeg artifacts, compression artifacts, blurry, artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits",
          "fewer digits, missing fingers, censored, bar censor, mosaic censoring, missing, extra, fewer, bad, hyper, error, ugly, worst, tagme, unfinished, bad proportions, bad perspective, aliasing",
          "simple background, asymmetrical, monochrome, sketch, concept art, flat color, flat colors, simple shading, jaggy lines, traditional media \(artwork\), microsoft paint \(artwork\), ms paint \(medium\)",
          "unclear, photo, icon, multiple views, sequence, comic, 2koma, 4koma, multiple images, turnaround, collage, panel skew, letterboxed, framed, border, speech bubble, 3d, lossy-lossless, scan artifacts",
          "out of frame, cropped,"
    ])
    
    NEG_ADDITIONAL = ", ".join([",(abstract:0.91), (doesnotexist:0.91)"])
    
    NEGATIVES : Dict[str, str] = {
        "Basic": NEG,
        "Extra": NEG_EXTRA,
        "Boost": NEG_BOOST
    }
    
    QUAILTY_BOOST = "masterpiece, best quality, good quality, very aesthetic, absurdres, newest, very awa, highres,"
    CINEMATIC = "(scenery, volumetric lighting, dof, depth of field)"
    
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
        "3:2 - (1536x1024)": "1536x1024"
    }
    
    SHOT_TYPES: Dict[str, str] = {
        "Close-Up": "close-up",
        "Medium Close-Up": "medium close-up",
        "Medium Shot": "medium shot",
        "Medium Long Shot": "medium long shot",
        "Long Shot": "long shot",
        "Wide Shot": "wide shot",
        "Portrait": "portrait",
        "Half-Body": "half-body",
        "Full-Body": "full-body",
        "Extreme Wide Shot": "extreme wide shot",
        "Extreme Close-Up": "extreme close-up",
        "Over-the-Shoulder": "over-the-shoulder",
        "Point of View": "point of view",
        "Establishing Shot": "establishing shot",
        "Aerial Shot": "aerial shot",
        "High Angle": "high angle",
        "Low Angle": "low angle",
        "Dutch Angle": "dutch angle",
        "Bird's Eye View": "bird's eye view",
    }


    DEFAULT = " ".join(
        [
            "leaning against a wall in a alley, glitch hair, iridescent hair, holding gun, profile, backlighting, ",
            "scenery of a ruin city, patrol team in the background"
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
                    {"forceInput": True, "tooltip": "Suffix to the prompt with a custom string."},
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
                    {"default": "2:3 - (1024x1536)", "tooltip": "Acts as a source filter."},
                ),
                "Batch Size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "Character": (["-"] + list(CHARACTERS.keys()), {"default": "-", "tooltip": "Danbooru Character List"}),
                "Artist": (["-"] + list(ARTISTS.keys()), {"default": "-", "tooltip": "Danbooru Artist List"}),
                "E621 Character": (["-"] + list(E621_CHARACTERS.keys()), {"default": "-", "tooltip": "E621 Character List"}),
                "E621 Artist": (["-"] + list(E621_ARTISTS.keys()), {"default": "-", "tooltip": "E621 Artist List"}),
                "Girl Characters": (["-"] + EasyNoobai.GIRLS, {"default": "-", "tooltip": "Number of Girl Characters"}),
                "Boy Characters": (["-"] + EasyNoobai.BOYS, {"default": "-", "tooltip": "Number of Boy Characters"}),
                "Mature Characters": ("BOOLEAN", {"default": False, "tooltip": "Changes Girl to Woman, Boy to Man"}),
                "Year": (["-"] + EasyNoobai.YEARS, {"default": "-", "tooltip": "Year of the character"}),
                "Shot Type": (["-"] + list(EasyNoobai.SHOT_TYPES.keys()), {"default": "-", "tooltip": "Type of shot"}),
                "Prompt": ("STRING", {"default": EasyNoobai.DEFAULT, "multiline": True}),
                "Negative Prompt": (
                    list(EasyNoobai.NEGATIVES.keys()),
                    {"default": "Basic", "tooltip": "Select the type of negative prompt to use."},
                ),
                "Format Tag": ("BOOLEAN", {"default": True, "tooltip": "Will format the tag using \( and \)"}),
                "Break Format": ("BOOLEAN", {"default": True, "tooltip": "Break the prompt into multiple tokens."}),
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
                "Prefix QB (Beta)": ("BOOLEAN", {"default": False, "tooltip": "Move the quality boost to the end of the prompt."}),
                ("Cinematic (Beta)"): ("BOOLEAN", {"default": False, "tooltip": "Add cinematic elements to the prompt."}),
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
    FUNCTION = "contruct"
    CATEGORY = "itsjustregi / Easy Noobai"
    
    def contruct(self, **kwargs) -> tuple:
        prompt_elements, negative_elements = [], []
        
        resolution = self.parse_resolution(kwargs["Resolution"])
        kwargs["Boy Characters"] is not "-" and prompt_elements.append(f"{kwargs['Boy Characters']},")
        kwargs["Girl Characters"] is not "-" and prompt_elements.append(f"{kwargs['Girl Characters']},")
        
        if kwargs["Format Tag"]:
            kwargs.get("Character") is not "-" and prompt_elements.append(f"\({kwargs['Character']}:1.2\),")
            kwargs.get("E621 Character") is not "-" and prompt_elements.append(f"\({kwargs['E621 Character']}:1.2\),")
            kwargs.get("Artist") is not "-" and prompt_elements.append(f"[[artist:{self.format_tag(kwargs['Artist'])}]],")
            kwargs.get("E621 Artist") is not "-" and prompt_elements.append(f"[[artist:{self.format_tag(kwargs['E621 Artist'])}]],")
        else:
            kwargs.get("Character") is not "-" and prompt_elements.append(f"{kwargs['Character']},")
            kwargs.get("E621 Character") is not "-" and prompt_elements.append(f"{kwargs['E621 Character']},")
            kwargs.get("Artist") is not "-" and prompt_elements.append(f"{self.format_tag(kwargs['Artist'])}")
            kwargs.get("E621 Artist") is not "-" and prompt_elements.append(f"{self.format_tag(kwargs['E621 Artist'])}")


        kwargs["SFW"] and prompt_elements.append("(sfw:1.2),")
        kwargs["Year"] is not "-" and prompt_elements.append(f"{kwargs['Year']},")
        kwargs["Shot Type"] is not "-" and prompt_elements.append(f"({kwargs['Shot Type']}:1.4),")
        kwargs.get("Prompt") and prompt_elements.append(kwargs["Prompt"])
        kwargs.get("suffix") and prompt_elements.append(kwargs["suffix"])

        # Construct negative elements
        
        if kwargs["Prefix QB (Beta)"]:
            kwargs["Quality Boost (Beta)"] and prompt_elements.insert(0,
                f"{self.QUAILTY_BOOST.strip()}{self.CINEMATIC.strip() if kwargs['Cinematic (Beta)'] else ''},"
            )
        else:
            kwargs["Quality Boost (Beta)"] and prompt_elements.append(
                f"{self.QUAILTY_BOOST.strip()}{self.CINEMATIC.strip() if kwargs['Cinematic (Beta)'] else ''},"
            )



        kwargs["SFW"] and negative_elements.append(self.CENSORSHIP.strip())
        
        final_prompt = " ".join(prompt_elements).lower()


        kwargs["Negative Prompt"] and negative_elements.append(
            self.NEGATIVES[kwargs["Negative Prompt"]]
        )
        
        if kwargs["Character"] is not "-" or kwargs["E621 Character"] is not "-":
            negative_elements.append(self.NEG_ADDITIONAL)
            
        if kwargs.get("Mature Characters", False):
            replacements = {
                "girls": "women",
                "girl": "woman",
                "boys": "men",
                "boy": "man"
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
        
        print(final_negative)
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
        dimensions = EasyNoobai.RESOLUTIONS[resolution].split('x')
        return int(dimensions[0]), int(dimensions[1])
    

    # add 'BREAK' keyword after every 70 characters making sure the last character is always a ","

    def format_prompt(self, prompt: str) -> str:
        # Remove all existing commas
        prompt = prompt.replace(',', '')
        
        # Split the prompt into words
        words = prompt.split()
        
        formatted_parts = []
        current_part = []
        current_length = 0

        for word in words:
            if current_length + len(word) > 70:
                formatted_parts.append(' '.join(current_part) + ' BREAK')
                current_part = [word]
                current_length = len(word)
            else:
                current_part.append(word)
                current_length += len(word) + 1  # +1 for the space

        # Add the last part
        if current_part:
            formatted_parts.append(' '.join(current_part))

        return ' '.join(formatted_parts)



    def format_tag(self, tag: str) -> str:
        formatted_tag = tag.replace('_', ' ')
        formatted_tag = formatted_tag.replace('(', r'\(').replace(')', r'\)')
        
        return formatted_tag

    # FROM COMFYUI CORE
    def generate(self, width, height, batch_size=1) -> tuple:
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )


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
                    {"forceInput": True, "tooltip": "Suffix to the prompt with a custom string."},
                ),
            },
            "required": {
                "Character (Base)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Character Base, Heavily influences the prompt"},
                ),
                "Character (Secondary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Character Secondary, influences the prompt"},
                ),
                "Character (Tertiary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Character Tertiary, influences the prompt less"},
                ),
                "Character (Quaternary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Character Quaternary, influences the prompt very little"},
                ),
                "Character (Quinary)": (
                    ["-"] + list(CHARACTERS.keys()),
                    {"default": "-", "tooltip": "Character Quinary, influences the prompt very little"},
                ),
                "Weighed Average": ("BOOLEAN", {"default": False, "tooltip": "Weighed Average of all characters"}),
                "Format Tag": ("BOOLEAN", {"default": True, "tooltip": "Will format the tag using \( and \)"}),
            },
        }
        
        return PSONA_UI_NOOBAI_CHARACTERS
    
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "CHARACTER PROMPT",
    )
    FUNCTION = "contruct"
    CATEGORY = "itsjustregi / Easy Noobai"
    
    def contruct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        
        characters = [
            ("Character (Base)", 1.4),
            ("Character (Secondary)", 1.3),
            ("Character (Tertiary)", 1.25),
            ("Character (Quaternary)", 1.15),
            ("Character (Quinary)", 1)
        ]

        character_prompts = []
        for char, weight in characters:
            char_value = kwargs.get(char)
            if char_value and char_value != "-":
                formatted_char = self.format_tag(char_value) if kwargs.get("Format Tag") else char_value
                if not prefix or formatted_char not in prefix:
                    character_prompts.append(f"{formatted_char}:{weight}")

        # Join character prompts with commas
        character_prompts_str = ", ".join(character_prompts)

        if prefix:
            def find_insertion_point(text):
                commas = list(re.finditer(r',', text))
                if len(commas) >= 3:
                    return commas[1].end()
                return len(text)

            insert_index = find_insertion_point(prefix)
            final_prompt = prefix[:insert_index].rstrip() + " " + character_prompts_str + ", " + prefix[insert_index:].lstrip()
        else:
            final_prompt = character_prompts_str

        final_prompt = re.sub(r'\s+', ' ', final_prompt).strip()

        if kwargs.get("Weighed Average", True):
            weighted_chars = re.findall(r'\([^)]+\):\d+(\.\d+)?', final_prompt)
            if weighted_chars:
                weighted_avg_result = self.weighted_avg(weighted_chars)
                for old, new in zip(weighted_chars, weighted_avg_result):
                    final_prompt = final_prompt.replace(old, new)
        
        if suffix:
            final_prompt += " " + suffix

        return (final_prompt,)

    @staticmethod 
    def weighted_avg(args: List[str]) -> List[str]:
        if not args:
            return []
        
        weights = []
        characters = []
        for arg in args:
            parts = arg.split(':')
            if len(parts) == 2:
                try:
                    character = parts[0]
                    weight = float(parts[1])
                    characters.append(character)
                    weights.append(weight)
                except ValueError:
                    print(f"Warning: Skipping invalid weight format in '{arg}'")
        
        if not weights:
            return args  # Return original list if no valid weights found
        
        num_characters = len(weights)
        if num_characters > 5:
            num_characters = 5  # Cap at 5 characters max
        
        total_weight = 1.2  # Total weight should be 1.2
        weight_per_character = total_weight / num_characters
        
        normalized_weights = [weight_per_character] * num_characters
        return [f"{char}:{weight:.4f}" for char, weight in zip(characters, normalized_weights)]


    @staticmethod
    def format_tag(tag: str) -> str:
        formatted_tag = tag.replace('_', ' ')
        formatted_tag = formatted_tag.replace('(', r'\(').replace(')', r'\)')
        return formatted_tag


    @staticmethod
    def add_prefix(prompt_elements: List[str], prefix: str) -> str:
        prompt = " ".join(prompt_elements).lower()
        gender_words = ['girl', 'girls', 'boy', 'boys', 'woman', 'women', 'man', 'men']
        pattern = r'\b(' + '|'.join(gender_words) + r')\b'
        
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            index = match.end()
            return prompt[:index] + " " + prefix + " " + prompt[index:]
        else:
            return prefix + " " + prompt
        
        
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
                    {"forceInput": True, "tooltip": "Suffix to the prompt with a custom string."},
                ),
            },
            "required": {
                "Artist (Base)": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Artist Base, Heavily influences the prompt"},
                ),
                "Artist (Secondary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Artist Secondary, influences the prompt"},
                ),
                "Artist (Tertiary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Artist Tertiary, influences the prompt less"},
                ),
                "Artist (Quaternary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Artist Quaternary, influences the prompt very little"},
                ),
                "Artist (Quinary)": (
                    ["-"] + list(ARTISTS.keys()),
                    {"default": "-", "tooltip": "Artist Quinary, influences the prompt very little"},
                ),
                "Weighed Average": ("BOOLEAN", {"default": False, "tooltip": "Weighed Average of all artists"}),
                "Format Tag": ("BOOLEAN", {"default": True, "tooltip": "Will format the tag using \( and \)"}),
            },
        }
        
        return PSONA_UI_NOOBAI_ARTISTS
    
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "ARTIST PROMPT",
    )
    FUNCTION = "contruct"
    CATEGORY = "itsjustregi / Easy Noobai"
    
    def contruct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        
        artists = [
            ("Artist (Base)", 1.4),
            ("Artist (Secondary)", 1.3),
            ("Artist (Tertiary)", 1.25),
            ("Artist (Quaternary)", 1.15),
            ("Artist (Quinary)", 1)
        ]

        artist_prompts = []
        for artist, weight in artists:
            artist_value = kwargs.get(artist)
            if artist_value and artist_value != "-":
                formatted_artist = self.format_tag(artist_value)
                if not prefix or formatted_artist not in prefix:
                    artist_prompts.append(f"({formatted_artist}):{weight}")


        artist_prompts_str = ", ".join(artist_prompts)

        if prefix:
            def find_insertion_point(text):
                weighted_strings = list(re.finditer(r'[^,]+:\d+(\.\d+)?', text))
                if weighted_strings:
                    return weighted_strings[-1].end()
                return 0  # If no weighted strings found, insert at the beginning

            insert_index = find_insertion_point(prefix)
            if insert_index > 0:
                # If we're not inserting at the beginning, add a comma before the new artists
                artist_prompts_str = ", " + artist_prompts_str
            final_prompt = prefix[:insert_index] + artist_prompts_str + prefix[insert_index:]
        else:
            final_prompt = artist_prompts_str

        final_prompt = re.sub(r'\s+', ' ', final_prompt).strip()

        if kwargs.get("Weighed Average", False):
            weighted_artists = re.findall(r'\([^)]+\):\d+(\.\d+)?', final_prompt)
            if weighted_artists:
                weighted_avg_result = self.weighted_avg(weighted_artists)
                for old, new in zip(weighted_artists, weighted_avg_result):
                    final_prompt = final_prompt.replace(old, new)
        
        if suffix:
            final_prompt += " " + suffix

        return (final_prompt,)


    @staticmethod 
    def weighted_avg(args: List[str]) -> List[str]:
        if not args:
            return []
        
        weights = []
        artists = []
        for arg in args:
            parts = arg.split(':')
            if len(parts) == 2:
                try:
                    artist = parts[0]
                    weight = float(parts[1])
                    artists.append(artist)
                    weights.append(weight)
                except ValueError:
                    print(f"Warning: Skipping invalid weight format in '{arg}'")
        
        if not weights:
            return args  # Return original list if no valid weights found
        
        num_artists = len(weights)
        if num_artists > 5:
            num_artists = 5  # Cap at 5 artists max
        
        total_weight = 1.2  # Total weight should be 1.2
        weight_per_artist = total_weight / num_artists
        
        normalized_weights = [weight_per_artist] * num_artists
        return [f"{artist}:{weight:.4f}" for artist, weight in zip(artists, normalized_weights)]


    @staticmethod
    def format_tag(tag: str) -> str:
        formatted_tag = tag.replace('_', ' ')
        formatted_tag = formatted_tag.replace('(', r'\(').replace(')', r'\)')
        return formatted_tag


    @staticmethod
    def add_prefix(prompt_elements: List[str], prefix: str) -> str:
        prompt = " ".join(prompt_elements).lower()
        gender_words = ['girl', 'girls', 'boy', 'boys', 'woman', 'women', 'man', 'men']
        pattern = r'\b(' + '|'.join(gender_words) + r')\b'
        
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            index = match.end()
            return prompt[:index] + " " + prefix + " " + prompt[index:]
        else:
            return prefix + " " + prompt

import re
from typing import List, Dict

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
                    {"forceInput": True, "tooltip": "Suffix to the prompt with additional tags."},
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
                "Lore Tags": ("STRING", {"default": "", "tooltip": "Additional lore tags"}),
                "General Tags": ("STRING", {"default": "", "tooltip": "General tags (min 10)"}),
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
            "Character (Tertiary)"
        ]

        character_tags = []
        for char in characters:
            char_value = kwargs.get(char)
            if char_value and char_value != "-":
                formatted_char = self.format_tag(char_value)
                character_tags.append(f"character:{formatted_char}")

        lore_tags = self.format_lore_tags(kwargs.get("Lore Tags", ""))
        general_tags = self.format_general_tags(kwargs.get("General Tags", ""))

        all_tags = character_tags + lore_tags + general_tags
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
        return re.sub(r'[^a-z0-9_]', '', tag.lower().replace(' ', '_'))

    @staticmethod
    def format_lore_tags(lore_tags: str) -> List[str]:
        return [f"lore:{NoobaiE621Characters.format_tag(tag)}" for tag in lore_tags.split(',') if tag.strip()]

    @staticmethod
    def format_general_tags(general_tags: str) -> List[str]:
        tags = [NoobaiE621Characters.format_tag(tag) for tag in general_tags.split(',') if tag.strip()]
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
                    {"forceInput": True, "tooltip": "Suffix to the prompt with additional tags."},
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
                "Artist Aliases": ("STRING", {"default": "", "tooltip": "Additional artist aliases"}),
                "Artist URLs": ("STRING", {"default": "", "tooltip": "Associated artist URLs"}),
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
        
        artists = [
            "Artist (Primary)",
            "Artist (Secondary)"
        ]

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
        return re.sub(r'[^a-z0-9_]', '', tag.lower().replace(' ', '_'))

    @staticmethod
    def format_aliases(aliases: str) -> List[str]:
        return [f"alias:{NoobaiE621Artists.format_tag(alias)}" for alias in aliases.split(',') if alias.strip()]

    @staticmethod
    def format_urls(urls: str) -> List[str]:
        return [f"url:{url.strip()}" for url in urls.split(',') if url.strip()]


