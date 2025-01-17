import os
from .src import (
    EasyNoobai,
    NoobaiCharacters,
    NoobaiArtists,
    NoobaiE621Characters,
    NoobaiE621Artists,
)

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

NODE_CLASS_MAPPINGS = {
    "EasyNoobai": EasyNoobai,
    "NoobaiCharacters": NoobaiCharacters,
    "NoobaiArtists": NoobaiArtists,
    "NoobaiE621Characters": NoobaiE621Characters,
    "NoobaiE621Artists": NoobaiE621Artists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyNoobai": "Noobai Prompt",
    "NoobaiCharacters": "Characters",
    "NoobaiArtists": "Artists",
    "NoobaiE621Characters": "E621 Characters",
    "NoobaiE621Artists": "E621 Artists",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
