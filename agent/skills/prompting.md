description: writing image prompts that the installed model families actually follow
# Prompting

SPECIFIC BEATS GENERIC. Name the subject, action, setting, light, framing and
style concretely. "A woman in a red raincoat crossing a wet neon street at night,
waist-up, shot on 35mm, shallow depth of field" beats "a cool city photo".

By model family
- Z-Image Turbo, Flux, Krea, Qwen-Image: plain descriptive sentences. 40-120 words.
  Put the subject first. Do not use weight syntax like (word:1.3). cfg stays 1.0
  for Z-Image Turbo; 8 steps is its design point.
- SDXL / SD1.5 checkpoints: comma-separated descriptive phrases work best; a short
  negative prompt (blurry, low quality, watermark, extra fingers) helps.
- Never put quotation marks around the whole prompt: many models render quoted
  words as visible lettering. Use quotes only for text that must appear in the image.

Counts and parts
- If the image has N people, panels or objects, state the number and describe
  each one on its own. "Two dogs: a black labrador on the left, a white poodle on
  the right" beats "some dogs".

Negative prompts
- Short and relevant. Do not negate something the user asked for.
