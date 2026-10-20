"""Title / caption / script generation.

Uses Claude (claude-haiku-4-5: cheap and fast, adequate for one-line copy)
when ANTHROPIC_API_KEY is set; otherwise, or on any API failure, falls back
to simple string templates so the pipeline still runs end to end.
"""
import json
import os
import random

MODEL = "claude-haiku-4-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "meme_caption": {"type": "string"},
        "narration_script": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "meme_caption", "narration_script", "tags"],
    "additionalProperties": False,
}

PROMPT = """Write copy for one vertical YouTube Short built from a clip of the anime "{anime}".
Mood of the clip: {mood}.

Return:
- title: under 70 characters, curiosity-driven, no clickbait lies, no hashtags
- meme_caption: one short relatable meme caption (under 12 words)
- narration_script: 3-4 sentences of spoken narration, first sentence is a strong hook, \
no spoiler-heavy plot details, no quotes from the show
- tags: 6-10 lowercase YouTube tags (no # symbol)"""


def _template(anime, mood):
    hype = mood == "hype"
    titles = ([f"{anime} just went crazy", f"This {anime} moment is insane",
               f"Nobody was ready for this {anime} scene"] if hype else
              [f"This {anime} scene hits different", f"The saddest {anime} moment",
               f"{anime} made everyone go quiet"])
    captions = (["when the playlist hits just right", "me after one coffee"] if hype else
                ["pov: it's 3am and you remember this scene", "they really did that to us"])
    script = (f"You are not ready for this moment from {anime}. "
              f"Everything built up to this exact scene. "
              f"Watch closely, because it changes everything. "
              f"Follow for more {anime}.") if hype else (
              f"This is the moment {anime} fans still can't talk about. "
              f"It came out of nowhere. "
              f"And somehow it still hurts every single time. "
              f"Follow for more {anime}.")
    slug = anime.lower().replace(" ", "")
    return {
        "title": random.choice(titles),
        "meme_caption": random.choice(captions),
        "narration_script": script,
        "tags": ["anime", slug, "shorts", "animeedit", mood],
        "source": "template",
    }


def generate(anime_name, mood="hype"):
    """Returns {title, meme_caption, narration_script, tags, source}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _template(anime_name, mood)
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT.format(anime=anime_name, mood=mood)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stop_reason={resp.stop_reason}")
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        data["title"] = data["title"][:100]  # YouTube's hard limit
        data["source"] = "claude"
        return data
    except Exception as e:  # noqa: BLE001 - any failure -> template fallback
        print(f"  [content_gen] Claude call failed ({type(e).__name__}: {e}); using template")
        return _template(anime_name, mood)


if __name__ == "__main__":
    import sys
    print(json.dumps(generate(sys.argv[1] if len(sys.argv) > 1 else "Blue Lock",
                              sys.argv[2] if len(sys.argv) > 2 else "hype"), indent=2))
