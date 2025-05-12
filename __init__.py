import os

from .src import (
    EasyNoobaiMasterModel,
    EasyNoobai,
    NoobaiCharacters,
    NoobaiArtists,
    NoobaiE621Characters,
    NoobaiE621Artists,
    NoobaiHairstyles,
    NoobaiClothing,
)

from .src.nodes.pony_tokens import NoobaiPony
from .src.nodes.poses import NoobaiPoses

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

NODE_CLASS_MAPPINGS = {
    "EasyNoobaiMasterModel": EasyNoobaiMasterModel,
    "EasyNoobai": EasyNoobai,
    "NoobaiCharacters": NoobaiCharacters,
    "NoobaiPoses": NoobaiPoses,
    "NoobaiHairstyles": NoobaiHairstyles,
    "NoobaiClothing": NoobaiClothing,
    "NoobaiArtists": NoobaiArtists,
    "NoobaiE621Characters": NoobaiE621Characters,
    "NoobaiE621Artists": NoobaiE621Artists,
    "NoobaiPony": NoobaiPony,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyNoobaiMasterModel": "Noobai Model",
    "EasyNoobai": "Noobai Prompt",
    "NoobaiPony": "Pony Tokens",
    "NoobaiHairstyles": "Hairstyles",
    "NoobaiClothing": "Clothing",
    "NoobaiCharacters": "Characters",
    "NoobaiPoses": "Poses",
    "NoobaiArtists": "Artists",
    "NoobaiE621Characters": "E621 Characters",
    "NoobaiE621Artists": "E621 Artists",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
