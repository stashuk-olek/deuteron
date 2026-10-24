# Deuteron three-minute video kit

Start with the [two-page recording script](recording-script.pdf) or its [editable Word version](recording-script.docx). The 359-word script covers six scenes, with narration ending at 2:55 and a five-second closing hold. Rehearse once; the [full presentation guide](presentation-script.md) has delivery cues, pronunciation notes, and optional sentence cuts.

Record one continuous take with clear speech. The [recording guide](recording-and-editing.md) explains camera setup and pauses. A basic talking-head video is enough: the slides and compiler demonstration can be inserted afterward. Keep background music out of the raw recording.

![Overview of the six slides](slides-overview.png)

| Asset | Use |
|---|---|
| [Recording script · PDF](recording-script.pdf) | Two printable pages with exact narration, timecodes, and short delivery cues. |
| [Recording script · Word](recording-script.docx) | Editable narration document. |
| [Full presentation guide](presentation-script.md) | Detailed visual directions, pronunciation, optional cuts, and claim boundaries. |
| [Narration only](narration.txt) | Paste into a teleprompter; headings and stage directions are omitted. |
| [Slides · PDF](slides.pdf) | Six fixed-layout slides for preview or presentation. |
| [Slides · PowerPoint](slides.pptx) | Editable text, diagrams, bars, and speaker notes with the timed narration. |
| [1080p slide images](slides/) | `slide-01.png` through `slide-06.png`, ready for video compositing. |
| [Real compiler demonstration](demo/README.md) | Portable command, actual emitted PTX, and capture provenance. |
| [Recording and editing guide](recording-and-editing.md) | One-take recording advice and the planned post-production treatment. |
| [Edit plan](edit-plan.json) | Machine-readable timeline, transition cues, caption-safe layout, music policy, and export settings. |
| [Claim sources](claims.json) | Verified architecture, hardware, test, and performance facts with source hashes. |

The story is: **what Deuteron compiles → how it works → hardware features → real instruction output → RTX 5090 results → reliability and scope**. The slides preserve the measured performance qualifiers and distinguish H100 regression tests from RTX 5090 feature examples.

## After recording

Keep the original video at its original quality. The next editing step will remove mistakes, align these slides to the actual speech, add restrained transitions and quiet licensed or original instrumental music, then generate captions from the edited recording. The plan includes clean and captioned 1080p exports. Source cut points, music, and caption timings are set after reviewing the recording.

The slides are complete still assets; no finished narrated video or aligned caption file is represented by this kit. Use the PDF or PNG files for fixed rendering. PowerPoint uses DejaVu Sans and Liberation Mono; installing those fonts preserves its intended editable layout.

## Rebuild the assets

The document renderer is separate from the compiler package. In a dedicated environment:

```sh
python -m pip install -r docs/video/requirements.txt
python docs/video/build_assets.py --output /tmp/deuteron-video-assets
```

The generator reads the timed script, checked-in measurements, and captured PTX. It never invokes the compiler or initializes a GPU. Linux font locations are listed in `build_assets.py`; adjust them when rebuilding on another system. Omitting `--output` regenerates the files in this directory, replacing generated assets.

The compiler and benchmark evidence correspond to release `efb728d`. Presentation additions leave the compiler, release measurements, original inputs, and handoff 9 unchanged. [assets-manifest.json](assets-manifest.json) records source and export hashes.
