import sys
from pathlib import Path
from typing import Dict
from enum import Enum


"""
EasyPony - Resources for implementation of EasyPony prompt sturcture
- https://civitai.com/articles/8547/prompting-for-score-or-source-or-rating-or-and-an-overview-of-prompting-syntax
- https://civitai.com/articles/4871/pony-diffusion-v6-xl-prompting-resources-and-info
- https://civitai.com/articles/4248/what-is-score9-and-how-to-use-it-in-pony-diffusion
- https://civitai.com/articles/6160/negative-prompt-for-pdxl-v2-works-with-other-models

"""


class NoobaiPony:

    class NoobaiPonyTokens(Enum):
        ONLY_THE_BEST = "score_9, highly detailed"
        GOOD = "score_9, score_8_up, score_7_up, highly detailed"
        AVERAGE = (
            "score_9, score_8_up, score_7_up, score_6_up, score_5_up, highly detailed"
        )
        EVERYTHING = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, highly detailed"

    SOURCES = [
        "source_anime",
        "source_pony",
        "source_fury",
        "source_cartoon",
    ]

    RATING = [
        "rating_safe",
        "rating_questionable",
        "rating_explicit",
    ]

    PONY_NEG = "text, deformed, bad hands, worst quality, low quality, deformed, censored, bad anatomy, watermark, signature,"

    PONY_CENSORSHIP = " ".join(
        [
            "bar censor, censor, censor mosaic, censored, filter abuse",
            "heavily pixelated, instagram filter, mosaic censoring, over filter",
            "over saturated, over sharpened, overbrightened, overdarkened",
            "overexposed, overfiltered, oversaturated",
        ]
    )

    PONY_QUALITY_BOOST = " ".join(
        [
            "ai-generated, artifact, artifacts, bad quality, bad scan, blurred",
            "blurry, compressed, compression artifacts, corrupted, dirty art scan",
            "dirty scan, dithering, downsampling, faded lines, frameborder, grainy",
            "heavily compressed, heavily pixelated, high noise, image noise, low dpi",
            "low fidelity, low resolution, lowres, moire pattern, moiré pattern",
            "motion blur, muddy colors, noise, noisy background, overcompressed",
            "pixelation, pixels, poor quality, poor lineart, scanned with errors",
            "scan artifact, scan errors, very low quality, visible pixels",
        ]
    )

    PONY_NEG_EXP = " ".join(
        [
            "3rd party watermark",
            "abstract, aliasing, alternate form, anatomically incorrect, artistic error",
            "asymmetrical, bad anatomy, bad aspect ratio, bad compression, bad cropping",
            "bad edit, bad feet, bad hands, bad leg, bad lighting, bad metadata, bad neck",
            "bad parenting, bad perspective, bad proportions, bad quality, bad shading",
            "bad trigger discipline, badly compressed, bar censor, black and white, black bars",
            "blur censor, blurred, blurry, broken anatomy, censor bar, censored, chromatic aberration",
            "color banding, color edit, color issues, compressed, compression artifacts, cropped",
            "deformed, depth of field, derivative work, disfigured, distracting watermark, downscaled",
            "edit, edited, edited screencap, elongated body, error, exaggerated anatomy, extra arms",
            "extra digits, extra fingers, extra legs, fused fingers, fused limbs, gif, gif artifacts",
            "greyscale, hair censor, has bad revision, has censored revision, has downscaled revision",
            "idw, incorrect leg anatomy, irl, jpeg artifacts, long neck, low quality, low res",
            "low resolution, lowres, md5 mismatch, meme, microsoft paint (software), missing arm",
            "missing body part, missing finger, missing leg, missing limb, mosaic censoring, ms paint",
            "mutated, mutation, mutilated, needs more jpeg, needs more saturation,",
            "novelty censor, obtrusive watermark, off-topic, photo, photoshop (medium), pixel art",
            "pixelated, pixels, recolor, resampling artifacts, resized, resolution mismatch, scan artifacts",
            "screencap, simple background, simple shading, sketch, source larger, source smaller, steam censor",
            "stitched, tail censor, third-party edit, third-party watermark, too many fingers, traditional art",
            "tumblr, typo, ugly, unfinished, upscaled, vector trace, wrong aspect ratio, wrong eye shape",
        ]
    )

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        PSONA_UI_PONY_TOKENS: Dict = {
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
                "Quality": (
                    list(NoobaiPony.NoobaiPonyTokens.__members__.keys()),
                    {
                        "default": "EVERYTHING",
                        "tooltip": "Quality of the image. i.e GOOD = score7-up",
                    },
                ),
                "Source": (
                    ["-"] + NoobaiPony.SOURCES,
                    {"default": "-", "tooltip": "Acts as a source filter."},
                ),
                "Rating": (
                    ["-"] + NoobaiPony.RATING,
                    {"default": "-", "tooltip": "Acts as a rating filter."},
                ),
                "Invert Source (Neg)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the source in the negative prompt.",
                    },
                ),
                "Invert Rating (Neg)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the rating in the negative prompt.",
                    },
                ),
                "Pony Quality Boost (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Boost the quality of the image using negative prompts.",
                    },
                ),
                "Pony Negative Boost (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Boost the negative aspects of the image using negative prompts.",
                    },
                ),
            },
        }

        return PSONA_UI_PONY_TOKENS

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "PROMPT",
        "NEGATIVE",
    )
    OUTPUT_IS_LIST = (
        False,
        False,
    )
    FUNCTION = "display"
    CATEGORY = "itsjustregi / Easy Noobai"

    def display(self, **kwargs) -> tuple:
        prompt_elements, negative_elements = [], []

        quality_value = f"{NoobaiPony.NoobaiPonyTokens[kwargs['Quality']].value}"
        source = "" if kwargs.get("Source") == "-" else kwargs["Source"]
        rating = "" if kwargs.get("Rating") == "-" else kwargs["Rating"]

        source_invert = kwargs.get("Invert Source (Neg)")
        rating_invert = kwargs.get("Invert Rating (Neg)")

        # Construct prompt elements
        quality_value and prompt_elements.append(
            ", ".join(
                filter(
                    None,
                    [
                        quality_value,
                        kwargs.get("prefix"),
                        f"{source}," if source and not source_invert else "",
                        f"{rating}," if rating and not rating_invert else "",
                    ],
                )
            )
        )

        kwargs.get("Prompt") and prompt_elements.append(kwargs["Prompt"])
        kwargs.get("suffix") and prompt_elements.append(kwargs["suffix"])

        # Construct negative elements
        kwargs["Pony Quality Boost (Beta)"] and negative_elements.append(
            self.PONY_QUALITY_BOOST.strip()
        )
        kwargs["Pony Negative Boost (Beta)"] and negative_elements.append(
            self.PONY_NEG_EXP.strip()
        )

        rating_invert and negative_elements.insert(0, f"{rating},")
        source_invert and negative_elements.insert(0, f"{source},")

        final_prompt = " ".join(prompt_elements).lower()
        final_negative = f"{self.PONY_NEG} {' '.join(negative_elements)}".lower()

        return (
            final_prompt,
            final_negative,
        )
