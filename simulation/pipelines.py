PIPELINES: dict[str, list[str]] = {
    "vignette":            ["persona", "validate_vignette"],
    "vignette_no_val":     ["persona"],
    "vignette_full":       ["craft_persona", "persona", "validate_vignette"],
    "zero_shot":           ["zero_shot"],
}
