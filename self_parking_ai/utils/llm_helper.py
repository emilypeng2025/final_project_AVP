# utils/llm_helper.py
import os

def _fallback_brief(strategy:str, entry_side:str, car_L:float, car_W:float,
                    spot_L:float, spot_W:float, inset:float, dist:float) -> str:
    return (
        f"Plan: {strategy}. Approach from the {entry_side} side.\n"
        f"- Car: {car_L:.2f} m × {car_W:.2f} m\n"
        f"- Spot: {spot_L:.2f} m × {spot_W:.2f} m\n"
        f"- Distance to spot: {dist:.2f} m; rear‑bumper inset: {inset:.2f} m\n"
        "Drive forward along the path, keep wheels smooth through the arcs, "
        "and stop when the rear bumper reaches the X marker."
    )

def summarize_plan_with_llm(strategy:str, entry_side:str, car_L:float, car_W:float,
                            spot_L:float, spot_W:float, inset:float, dist:float) -> str:
    """
    Returns a short friendly driving brief. Uses OpenAI if OPENAI_API_KEY is set,
    otherwise returns a local fallback string.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # No key: offline summary
        return _fallback_brief(strategy, entry_side, car_L, car_W, spot_L, spot_W, inset, dist)

    try:
        # OpenAI official Python SDK (Responses API or Chat Completions—both supported)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are a concise driving assistant. Given a parking plan, produce a 4–6 line, "
            "clear, calm set of instructions (no extra chit‑chat). Include approach side, "
            "key turns, and stopping cue.\n\n"
            f"Strategy: {strategy}\n"
            f"Entry side: {entry_side}\n"
            f"Car L×W: {car_L:.2f} × {car_W:.2f} m\n"
            f"Spot L×W: {spot_L:.2f} × {spot_W:.2f} m\n"
            f"Distance to spot: {dist:.2f} m\n"
            f"Rear‑bumper inset: {inset:.2f} m\n"
        )

        # Prefer Responses API; Chat Completions also ok.
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            instructions="Be direct and helpful. No emojis.",
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # Very old clients may not have output_text — try choices access (chat style)
            # but we’ll just fall back if it’s not there.
            return _fallback_brief(strategy, entry_side, car_L, car_W, spot_L, spot_W, inset, dist)
        return text.strip()

    except Exception:
        # Any error → safe fallback
        return _fallback_brief(strategy, entry_side, car_L, car_W, spot_L, spot_W, inset, dist)
