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
        "Profile": "profile",
        "Upper Body": "upper body",
        "Lower Body": "lower body",
        "On Back": "on back, inverted",
        "Feet Out Of Frame": "feet out of frame",
        "Cowboy Shot": "cowboy shot",
        "Full Body": "full body",
        "Wide Shot": "wide shot",
        "Very Wide ": "very wide",
        "Cropped Arms": "cropped arms",
        "Cropped Legs": "cropped legs",
        "Cropped Shoulders": "cropped shoulders",
        "Cropped Head": "cropped head",
        "Cropped Torso": "cropped torso",
        "Close-up": "close-up",
        "Cut-in": "cut-in",
        "Split crop": "split crop",
        "Multiple Views": "multiple views"
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
    
    # Based on Civitai Article https://civitai.com/articles/8804/illustrious-xl-noobai-xl-hairstyles
    # by https://civitai.com/user/lizardon1024
    HAIRSTYLE: Dict[str, Dict[str, str]] = {
        "Length and Volume": ["very short hair", "short hair", "medium hair", "long hair", "very long hair", "absurdly long hair", "big hair", "bald girl"],
        "Haircuts": {
            "Short": ["bob cut", "inverted bob", "bowl cut", "buzz cut", "pixie cut", "undercut"],
            "Medium": ["flipped hair"],
            "Long": ["hime cut"]
        },
        "Hairstyles": {
            "Tied": ["bow-shaped hair", "flower-shaped hair", "hair updo", "one side up", "two side up", "low-tide long hair", "multi-tied hair", "twintails", "low twintails", "short twintails", "twisted hair"],
            "Braids": ["front braid", "side braid", "french braid", "single braid", "twin braids", "half up braid", "low-braided long hair", "cornrows", "dreadlocks"],
            "Hair buns": ["braided bun", "single hair bun", "double bun", "cone hair bun", "doughnut hair bun"],
            "Hair rings": ["hair rings", "single hair ring"],
            "Ponytails": ["ponytails", "folded ponytail", "front ponytail", "high ponytail", "short pontail", "side ponytail", "topknot"],
            "Tall hair": ["afro", "beehive hairdo", "crested hair", "pompadour"],
            "Hair texture": ["wavy hair", "straight hair", "spiked hair", "ringlets", "pointy hair", "messy hair", "hair flaps", "twin drills", "drill hair", "curly hair"],
        },
        "Hairstyle Front": {
           "Bangs": ["bangs", "arched bangs", "asymmetrical bangs", "bangs pinned back", "blunt bangs", "crossed bangs", "diagonal bangs", "hair over eyes", "hair over one eye", "long bangs", "parted bangs", "curtained hair", "wispy bangs", "short bangs", "hair between eyes"],
            "Hair intakes": ["hair intakes", "single hair intake"],
            "Sidelocks": [ "sidelocks", "asymmetrical sidelocks", "drill sidelocks", "low-tied sidelocks", "single sidelocks", "widow\'s peak"]
        },
        "Hairstyle Top": {
            "Top of the head": ["ahoge", "heart ahoge", "huge ahoge", "antenna hair", "heart antenna hair", "hair pulled back", "hair slicked back", "mohawk"],
        },
        "Hair Colors": {
            "Aqua Hair": ["aqua hair", "dark aqua hair", "light aqua hair"],
            "Black Hair": ["black hair", "multicolred black hair", "gradient black hair"],
            "Blonde Hair": ["blonde hair", "multicolred blonde hair", "gradient blonde hair"],
            "Blue Hair": ["blue hair", "dark blue hair", "light blue hair"],
            "Light Blue Hair": ["light blue hair", "multicolred light blue hair", "gradient light blue hair"],
            "Dark Blue Hair": ["dark blue hair", "multicolred dark blue hair", "gradient dark blue hair"],
            "Brown Hair": ["brown hair", "dark brown hair", "light brown hair"],
            "Light Brown Hair": ["light brown hair", "multicolred light brown hair", "gradient light brown hair"],
            "Green Hair": ["green hair", "dark green hair", "light green hair"],
            "Grey Hair": ["grey hair", "dark grey hair", "light grey hair"],
            "Orange Hair": ["orange hair", "dark orange hair", "light orange hair"],
            "Pink Hair": ["pink hair", "dark pink hair", "light pink hair"],
            "Purple Hair": ["purple hair", "dark purple hair", "light purple hair"],
            "Red Hair": ["red hair", "dark red hair", "light red hair"],
            "White Hair": ["white hair", "dark white hair", "light white hair"],
            "Colored Inner Hair": ["colored inner hair", "dark colored inner hair", "light colored inner hair"],
            "Colored Tips": ["colored tips", "dark colored tips", "light colored tips"],
            "Roots": ["roots", "dark roots", "light roots"],
            "Gradient Hair": ["gradient hair", "dark gradient hair", "light gradient hair"],
            "Patterned Hair": ["patterned hair", "dark patterned hair", "light patterned hair"],
            "Rainbow Hair": ["rainbow hair", "dark rainbow hair", "light rainbow hair"],
            "Split-Color Hair": ["split-color hair", "dark split-color hair", "light split-color hair"],
            "Spotted Hair": ["spotted hair", "dark spotted hair", "light spotted hair"],
            "Streaked Hair": ["streaked hair", "dark streaked hair", "light streaked hair"],
            "Striped Hair": ["striped hair", "dark striped hair", "light striped hair"],
            "Raccoon Tails": ["raccoon tails", "dark raccoon tails", "light raccoon tails"],
            "Two-Tone Hair": ["two-tone hair", "dark two-tone hair", "light two-tone hair"]
        }
    }
    
    CLOTHING: Dict[str, Dict[str, List[str]]] = {
        "outfits": [
            "tuxedo", "evening_gown", "canonicals", "cocktail_dress", "gown",
            "wedding_dress", "maid", "miko", "school_uniform", "sailor",
            "serafuku", "sailor_senshi_uniform", "summer_uniform", "naval_uniform", "military_uniform",
            "business_suit", "nurse", "chef_uniform", "labcoat", "cheerleader",
            "band_uniform", "space_suit", "leotard", "domineering", "cheongsam",
            "china_dress", "chinese_style", "traditional_clothes", "uchikake", "off-shoulder_dress",
            "sleeveless_kimono", "print_kimono", "japanese_clothes", "hanten_(clothes)", "hanbok",
            "korean_clothes", "german_clothes", "gothic", "lolita", "gothic_lolita",
            "byzantine_fashion", "tropical cloth", "indian_style", "Ao_Dai", "ainu_clothes",
            "arabian_clothes", "egyptian_clothes", "hawaii costume", "furisode", "animal_costume",
            "bunny_costume", "cat_costume", "dog_costume", "bear_costume", "santa_costume",
            "hoodie", "homewear", "pajamas", "nightgown", "sleepwear",
            "print_pajamas", "yukata", "chinese_clothes", "hanfu", "Taoist robe",
            "halloween_costume", "kourindou_tengu_costume", "meme_attire", "casual", "loungewear",
            "robe", "cloak", "hooded_cloak", "winter_clothes", "down jacket",
            "santa", "harem_outfit", "shrug_clothing", "gym_uniform", "athletic_leotard",
            "volleyball_uniform", "tennis_uniform", "baseball_uniform", "letterman_jacket", "biker_clothes",
            "bikesuit", "wrestling_outfit", "front_zipper_swimsuit", "shell_bikini", "frilled_swimsuit",
            "strapless_dress", "backless_dress", "halter_dress", "sundress", "sleeveless_dress",
            "sailor_dress", "summer_dress", "pinafore_dress", "frilled_dress", "sweater_dress",
            "armored_dress", "fur-trimmed_dress", "lace-trimmed_dress", "collared_dress", "layered_dress",
            "pleated_dress", "taut_dress", "pencil_dress", "multicolored_dress", "striped_dress",
            "checkered_skirt", "polka_dot_dress", "plaid_dress", "print_dress", "vertical-striped_dress",
            "ribbed_dress", "short_jumpsuit", "multicolored_clothes", "expressive_clothes", "multicolored_bodysuit"
        ],
        "top": [
            "skirt", "mini_skirt", "skirt_suit", "bikini_skirt", "pleated_skirt",
            "pencil_skirt", "bubble_skirt", "tutu", "ballgown", "beltskirt",
            "denim_skirt", "suspender_skirt", "long_skirt", "summer_long_skirt", "hakama_skirt",
            "high-waist_skirt", "suspender_long_skirt", "chiffon_skirt", "lace_skirt", "ribbon-trimmed_skirt",
            "layered_skirt", "print_skirt", "multicolored_skirt", "striped_skirt", "plaid_skirt",
            "flared_skirt", "floral_skirt", "jumpsuit", "hot_pants", "striped_shorts",
            "suspender_shorts", "denim_shorts", "puffy_shorts", "dolphin_shorts", "dolfin_shorts",
            "tight_pants", "track_pants", "yoga_pants", "bike_shorts", "gym_shorts",
            "pants", "puffy_pants", "pumpkin_pants", "hakama_pants", "harem_pants",
            "bloomers", "buruma", "jeans", "cargo_pants", "camouflage_pants",
            "capri_pants", "chaps", "lowleg_pants", "plaid_pants", "single_pantsleg",
            "striped_pants", "torn_jeans", "hakama", "harness", "rigging"
        ],
        "bottoms": [
            "blouse", "collared_shirt", "dress_shirt", "sailor_shirt", "cropped_shirt",
            "t-shirt", "off-shoulder_shirt", "shrug_clothing", "gym_shirt", "cardigan",
            "criss-cross_halter", "frilled_shirt", "sweatshirt", "hawaiian_shirt", "hoodie",
            "kappougi", "plaid_shirt", "polo_shirt", "print_shirt", "sleeveless_hoodie",
            "sleeveless_shirt", "striped_shirt", "tank_top", "vest", "waistcoat",
            "tied_shirt", "undershirt", "crop_top", "camisole", "midriff",
            "oversized_shirt", "borrowed_garments", "blazer", "overcoat", "double-breasted",
            "long_coat", "winter_coat", "hooded_coat", "fur_coat", "fur-trimmed_coat",
            "duffel_coat", "parka", "cropped_jacket", "track_jacket", "hooded_track_jacket",
            "military_jacket", "camouflage_jacket", "leather_jacket", "letterman_jacket", "fur_trimmed_jacket",
            "two-tone_jacket", "trench_coat", "windbreaker", "raincoat", "hagoromo",
            "tunic", "cape", "capelet", "winter_clothes", "sweater",
            "pullover_sweaters", "ribbed_sweater", "sweater_vest", "backless_sweater", "aran_sweater",
            "beige_sweater", "brown_sweater", "hooded_sweater", "off-shoulder_sweater", "striped_sweater",
            "puffer_jacket", "short_over_long_sleeves", "impossible_clothes", "heart_cutout", "ofuda_on_clothes",
            "waist_apron", "maid_apron", "waist_cape", "clothes_around_waist", "jacket_around_waist",
            "sweater_around_waist", "loincloth", "bustier", "corset", "girdle",
            "armor", "bikini_armor", "full_armor", "plate_armor", "japanese_armor",
            "kusazuri", "power_armor", "mecha", "helmet", "kabuto",
            "off-shoulder_armor", "shoulder_armor", "muneate", "breastplate", "faulds",
            "wringing_clothes", "shiny_clothes", "kariginu", "front-tie_top", "jacket_on_shoulders"
        ]
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
                        "max": 100,
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

        kwargs["Negative Prompt"] != "-" and negative_elements.append(
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


class NoobaiHairstyles:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        HAIRSTYLE_UI: Dict = {
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
            "Length and Volume": (
                ["-"] + list(EasyNoobai.HAIRSTYLE["Length and Volume"]),
                {"default": "-", "tooltip": "Select a Length and Volume"},
            ),
            "Haircuts": (
                ["-"] + [
                f"{category.lower()} {option}"
                for category, options in EasyNoobai.HAIRSTYLE["Haircuts"].items()
                for option in options
                ],
                {"default": "-", "tooltip": "Select a haircut"},
            ),
            "Hairstyles": (
                ["-"] + [
                f"{category.lower()} {option}"
                for category, options in EasyNoobai.HAIRSTYLE["Hairstyles"].items()
                for option in options
                ],
                {"default": "-", "tooltip": "Select a hairstyle"},
            ),
            "Hair Colors": (
                ["-"] + [
                f"{option}"
                for category, options in EasyNoobai.HAIRSTYLE["Hair Colors"].items()
                for option in options
                ],
                {"default": "-", "tooltip": "Select a Hair Color"},
            ),
            "Inject Styles": (
                "BOOLEAN",
                {"default": True, "tooltip": "Inject into prefix content"},
            ),
            "Format Tag": (
                "BOOLEAN",
                {"default": False, "tooltip": "Will format the tag using \( and \)"},
            ),
            },
        }

        return HAIRSTYLE_UI

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        
        # Get hairstyle components
        length_vol = kwargs.get("Length and Volume", "-")
        haircut = kwargs.get("Haircuts", "-")
        hairstyle = kwargs.get("Hairstyles", "-")
        haircolors = kwargs.get("Hair Colors", "-")
        inject_styles = kwargs.get("Inject Styles", True)
        format_tag = kwargs.get("Format Tag", False)
        
        # Filter valid components
        components = [comp for comp in [length_vol, haircut, hairstyle, haircolors] if comp != "-"]
        
        if not components or not inject_styles :
            return (f"{prefix} {suffix}".strip(),)
        
        # Create combined hairstyle string
        combined = ", ".join(components)
        formatted = f"\({combined}\)" if format_tag else combined
        
        # Handle prefix insertion logic
        if prefix:
            # Split prefix into maximum 6 parts (5 commas)
            parts = prefix.split(',', 5)
            
            if len(parts) > 5:
                # Insert after 5th comma if exists
                new_prefix = f"{','.join(parts[:5])}, {formatted},{parts[5]}"
            else:
                # Append to end if fewer than 5 commas
                new_prefix = f"{prefix}, {formatted}"
            
            prompt = f"{new_prefix} {suffix}".strip().replace(" ,", ",")
        else:
            # No prefix case
            prompt = f"{formatted} {suffix}".strip()
        
        return (prompt,)

    
class NoobaiClothing:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        CLOTHING_UI: Dict = {
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
                "Outfits": (
                    ["-"] + EasyNoobai.CLOTHING["outfits"],
                    {"default": "-", "tooltip": "Select an outfit."},
                ),
                "Top": (
                    ["-"] + EasyNoobai.CLOTHING["top"],
                    {"default": "-", "tooltip": "Select a top clothing item."},
                ),
                "Bottoms": (
                    ["-"] + EasyNoobai.CLOTHING["bottoms"],
                    {"default": "-", "tooltip": "Select a bottom clothing item."},
                ),
                "Inject Styles": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Inject into prefix content."},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Will format the tag using \( and \)."},
                ),
            },
        }

        return CLOTHING_UI

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "itsjustregi / Easy Noobai"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        # Get clothing components
        outfit = kwargs.get("Outfits", "-")
        top = kwargs.get("Top", "-")
        bottoms = kwargs.get("Bottoms", "-")
        inject_styles = kwargs.get("Inject Styles", True)
        format_tag = kwargs.get("Format Tag", False)

        # Filter valid components
        components = [comp for comp in [outfit, top, bottoms] if comp != "-"]

        if not components or not inject_styles:
            return (f"{prefix} {suffix}".strip(),)

        # Create combined clothing string
        combined = ", ".join(components)
        formatted = f"\({combined}\)" if format_tag else combined

        # Handle prefix insertion logic
        if prefix:
            parts = prefix.split(',', 8)

            if len(parts) > 8:
                # Insert after 5th comma if exists
                new_prefix = f"{','.join(parts[:8])}, {formatted},{parts[8]}"
            else:
                # Append to end if fewer than 8 commas
                new_prefix = f"{prefix}, {formatted}"

            prompt = f"({new_prefix}1.6 {suffix}1.6)".strip().replace(" ,", ",")
        else:
            # No prefix case
            prompt = f"({formatted}1.6 {suffix}1.6)".strip()

        return (prompt,)