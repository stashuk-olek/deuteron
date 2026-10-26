# Recording and editing the Deuteron video

The target is a three-minute presentation. Record one continuous talking-head take using the supplied script; the edit will insert nine full-screen slides focused on compiler passes and engineering contributions. Editing is **awaiting your recording**.

## Record one clean take

- Use landscape **1080p, 16:9, 30 fps** when available. Put the camera at eye level, frame your head and shoulders, and use a clean background with light facing you.
- Record clear, dry speech in a quiet room. Keep the microphone reasonably close; avoid clipping or strong room echo. A short playback check before the full take is enough.
- Keep music out of the recording. Speak naturally and leave **2–3 seconds of silence before the first sentence and after the last**. Brief pauses between sections help the edit.
- If you make a mistake, pause for 2–3 seconds and restart the whole sentence. Keep recording; the earlier attempt can be removed.
- Rehearse once. Aim to finish the spoken content by roughly 2:55 in the edited video. Extra silence and restarted sentences may make the raw take longer than three minutes.

The only required asset is the original video file with your voice. Keep its original quality and filename; editing will use a copy. If already available, a separate microphone recording, a short screen capture, or a note identifying the preferred take is useful. None is required. For separately recorded sound, one visible clap near the start makes synchronization easier.

## Planned edit

These are target positions in the finished video, not cut points in your raw recording. Slide IDs connect this guide to [edit-plan.json](edit-plan.json); the presentation script supplies each scene's content.

| Time | Slide | Picture |
| --- | --- | --- |
| 0:00–0:12 | `slide-01` | The compiler transformations we built |
| 0:12–0:27 | `slide-02` | Where our passes run |
| 0:27–0:44 | `slide-03` | Reuse data before loading it again |
| 0:44–1:03 | `slide-04` | Reshape parallel work |
| 1:03–1:26 | `slide-05` | Turn contractions into tensor work |
| 1:26–1:49 | `slide-06` | Rewrite the pipeline feeding tensor cores |
| 1:49–2:12 | `slide-07` | Recognize and stage the attention recurrence |
| 2:12–2:30 | `slide-08` | Redistribute probability–value work |
| 2:30–2:55 | `slide-09` | Measure the whole compiler on RTX 5090 |
| 2:55–3:00 | `slide-09` | Static RTX 5090 comparison; no speech or music |

The three optional pass-reference slides are for preparation and discussion, outside the three-minute video. The prior standalone PTX demonstration and reliability/test-count slides are removed from the narrated deck.

Use clean cuts and occasional **6–8-frame crossfades** at 30 fps, with restrained motion. Keep slides readable; add a small presenter inset only where it leaves the slide and captions clear. Use the pass diagrams for the indicated intervals; there is no separate terminal-demo segment in this revision. Match changes to the actual spoken phrases, then adjust pauses and inserts to approach three minutes. If the narration overruns, shorten a sentence with a pickup rather than cutting words or speeding up the voice.

## Sound, captions, and delivery

Add a quiet instrumental ambient or electronic track with no vocals during editing. Use original or properly licensed music and retain its title, creator, source or creation record, license/rights evidence, and required attribution with the project. No music has been selected or generated yet.

Keep speech dominant: start with music about 18–24 dB below the voice and duck it further during speech as needed. **−16 LUFS integrated and a −1 dBTP ceiling are initial mix targets**, to be measured and adjusted after listening to the actual recording and encoded export. Music fades and transitions follow the final voice timing; all audio ends before the five-second silent hold.

For the captioned export, scale each complete slide to 1728 × 972, centered horizontally, and reserve the bottom 108 pixels for captions. This keeps chart notes and code visible. The clean export can use the full 1920 × 1080 slide.

Transcribe the actual recording, correct technical names and numbers, and align captions to the edited audio after the cuts settle. No final SRT exists before that alignment. Deliver both a clean master and a captioned master, plus SRT/VTT sidecars: **H.264/AAC, 1920×1080, constant 30 fps**, with 48 kHz audio. Preserve the original recording, editable project, music provenance, and exports separately.
