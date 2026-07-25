# Metis brand assets

The mark is a tetradrachm: a filled field, a beaded rim, and the mu of ΜΗΤΙΣ punched into it. Athenian
silver is the reference, currency being the first instrument anyone agreed to trust because it was
verifiably struck to a standard. The wordmark sits beside it at cap height, and each repository names
itself after it: `METIS / BENCHMARK`, `METIS / API`, and so on.

| File | Use |
| --- | --- |
| `metis-mark.svg` | the mark at 24px and up, where the rim reads |
| `metis-mark-small.svg` | below 24px: the rim is dropped, the disc and the mu carry it alone |
| `metis-mark-bronze.svg` | the bronze field |
| `metis-mark-outline.svg` | one keyline on paper |
| `metis-mark-reversed.svg` | on an ink ground |
| `metis-lockup.svg` / `-reversed.svg` / `-bronze.svg` | mark, wordmark and the line beneath it |
| `metis-favicon-512.png` | favicon, avatar, app icon |
| `metis-tokens.css` | colour, type and tracking tokens |

Every SVG here is self-contained: the letterforms are outlined, so the mark renders the same whether or
not Bodoni Moda and Cormorant Garamond are installed.

**Clear space** equals the rim inset, r/6, on every side. **Minimum size** is 24px for the full mark;
below that use `metis-mark-small.svg`.

**Type.** Bodoni Moda for titles and verdict statements, whose high-contrast strokes echo the punched
mu. Cormorant Garamond for the wordmark only, always letterspaced 0.16em, never body copy. IBM Plex Sans
for body and UI, IBM Plex Mono for every number, metric, dataset and command.

**Colour.** Ink `#14181A` and a cool neutral ground carry almost every surface. Bronze `#2C5449` marks
one thing per view: the reading that matters. The verdict colours, brass `#9A7B3F` for under-powered
evidence and oxblood `#7B2B33` for a fired stop rule, are reserved for pre-registered outcomes and are
never used decoratively.

**Don't.** No metallic gradient, no emboss, no drop shadow: it is a flat strike, not a rendering of
silver. Do not set the wordmark without its tracking, and do not use the mu on its own as a letter.

## One note on the mu

Bodoni Moda ships no U+039C, so a live Greek mu falls back to Georgia and the mark loses the
high-contrast strokes the type spec describes. Greek capital mu and Latin M are the same letterform, so
the outlines here are drawn from Bodoni's M at a size that reproduces the cap height of the original,
which keeps the intended typeface and the intended proportions at once.
