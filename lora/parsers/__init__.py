"""Parser registry entry point."""
from . import fl, starnodes, comfy_native, hotstep, lokr, peft, artist_bundle, mothersuperior


def register_all():
    from ..detect import register_parser
    for name, parser in (("fl", fl.parse), ("starnodes", starnodes.parse),
                         ("comfy_native", comfy_native.parse),
                         ("hotstep_fused", hotstep.parse_fused),
                         ("hotstep_native", hotstep.parse_native),
                         ("hotstep_lokr", lokr.parse),
                         ("peft", peft.parse),
                         ("mothersuperior", mothersuperior.parse),
                         ("artist_bundle", artist_bundle.parse)):
        register_parser(name, parser)

register_all()
