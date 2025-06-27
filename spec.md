
## File structure:

### spec.md:

- this file
- describes the functionality in full, including examples of encoding, training, tokenizing
- includes code style instructions and desired formats
- reference this always!

### train.py: 

- entry point, glue code
- handles loading and training with unsloth
- sets up datasets
- keep it simple and training-focused, import if possible

### data.py

- handles the datasets and collator
- also handles the token space conversion
- wraps other functions/classes for handling audio and maps directly
- does caching during training

### maps.py

- handles parsing beat saber maps
- utils for zip files, v2/v3 interop, and map json

### audio.py

- handles audio encoding with encodec

### infer.py

- also entry point, glue code
- handles model loading, inference, and decoding
- like train, keep it simple and import if possible

### utils.py (new)

- handles basic utility functions
- like converting special v3 notes (arcs, chains) to regular notes
- or aligning notes to the PPQ grid
- or creating ready-to-play .zip files out of the inferred output
- eventually, will handle some scraping (ytdlp for getting song files, for example)

### scrape.py (new)

- separate from all other functionality in this program (except imports some stuff from maps.py)
- handles scraping beatsaver to get maps
- includes filtering code to exclude maps that don't meet certain criteria
- handles map metadata in case of future training runs
- filtering code should run on its own, so it can filter from existing downloaded maps

Do not create any additional files, all necessary functionality is covered in these.
-----

## comment schema and code style

comments should describe the data at each point instead of jumping around the code to figure out what everything is. 

it should be described in the functions that use them:

```py
    def __init__(self, base_dataset, chunk_duration=30.0, max_seq_length=32767):

        # base_dataset is a ZippedBeatSaberDataset (directory of zip files passed to it)
        self.base = base_dataset 

    ...

        # audio_chunk = (frames, 2)
        # notes_chunk = [(time, note_type), ...]
        return create_training_sequence(audio_chunk, notes_chunk)
```

instead of describing irrelevant obvious functionality (`# Calculate beats in chunk`).

Keep functions and classes clearly defined and kept in their respective files.
for instance, only data.py should handle the special token space - if other files really need it,
they can import, but they really shouldn't. 
anything that has to do with parsing maps should never be anywhere except maps.py.
caching glue code should not be in train.py, since it belongs in data.py

code should always be simple and minimal, and the functionality should be clear from the code itself. comments are for tracking data flow, not math.

use data.py as the example of good clean code in the style you should be replicating.

DO NOT change any parts of the code you are not explicitly requested to. Doing that has a good chance
of introducing breaking changes elsewhere (ie, changing training params, sequence lengths,
data formats, etc). Those decisions were made on purpose. Keep your edits surgical. 

-----

the important bits:

- training code: simple, follow the given setup for loading and training the model with unsloth
- audio encoding: also simple, use encodec exactly as shown
- tokenizing: follow the given token space. use the map parser and audio parser to get data to map to tokens
- dataloader: one complicated one that handles the map parsing in maps.py, and  tne simpler one
in data.py that handles the tokenizing, chunking, and caching
- chunking: aligns song tokens and map tokens (according to beat value) and returns a tuple with the associated song/map tokens for a given length. (eg 25-55 seconds, beats 50-110 @ 120 bpm)
- caching: keep it simple. it should store temp files somewhere - full length. cache on first encounter, then load from cache if exists. if possible, it should fit inside BeatDataset.py as a functionality extension, rather than adding a *third* dataloader wrapper. Clearly separate caching from the rest of what the dataloader does.
- data collator: set the labels for the *audio tokens* to -100, so the model only learns to generate the note tokens. special tokens should also not be trained on (except EOS), they will be prefilled. Specifically, *only map tokens and EOS will be trained on*
- map encoding: lists and unzips zip files, parses the .dat files (Info.dat and various difficulties) as json. Handles both v2 and v3 style maps. Parses info like BPM out of info.dat. handles note encoders/decoders
- infer: loads and tokenizes a song, prefills special tokens, runs inference, decodes notes, creates
output json

ignore utils and scraping for now. those are special. 

-----

the spec itself:

### Training Sequence Format
```
[BOS] [AUDIO_START] audio_tokens... [AUDIO_END] [NOTES_START] note_tokens... [EOS] [PAD...]
```

### Audio Section
- Flattened Encodec tokens: `[s0, s1, s0, s1, ...]`
- 1024 possible values per codebook, 2 codebooks per frame
- Each frame represents ~10ms of audio
- 30-second audio sequence -> ~6000 tokens

### Notes Section  
- Interleaved time/type pairs: `[time1, type1, time2, type2, ...]`
- Beat times are converted to ticks: `tick = int(round(beat * PPQ))`, PPQ = 48
- Times relative to audio chunk start
- Note type: `note_type = (y * 3 + x) * 20 + color * 10 + direction`
Where:
- `x ∈ {0,1,2}` (left, center, right)
- `y ∈ {0,1,2}` (bottom, middle, top) 
- `color ∈ {0,1}` (red, blue)
- `direction ∈ {0,1,2,3,4,5,6,7,8,9}` (cut directions)

## Token Space

```
Range                    | Purpose                           | Size      | Offset
------------------------|-----------------------------------|-----------|--------
0 … 12,287              | Time tick IDs (48 PPQ)           | 12,288    | 0
12,288 … 12,467         | Note type IDs                     | 180       | 12,288  
12,468 … 14,515         | Audio tokens (2 × 1024)          | 2,048     | 12,468
14,516                  | <BOS> (Beginning of Sequence)    | 1         | 14,516
14,517                  | <EOS> (End of Sequence)          | 1         | 14,517
14,518                  | <PAD> (Padding)                   | 1         | 14,518
14,519                  | AUDIO_START (Separator)           | 1         | 14,519
14,520                  | AUDIO_END (Separator)             | 1         | 14,520
14,521                  | NOTES_START (Separator)           | 1         | 14,521
```

Audio tokens are masked during training, so it learns to only predict note tokens given prepended audio
Inference prefills BOS, audio tokens, AUDIO_END, and NOTES_START

## Beat Saber Map Format:

beat saber maps are .zip files containing the following:
```
map.zip
├── Info.dat          # {"_beatsPerMinute": 120, ...}
├── song.egg          # audio file (ogg/mp3)
├── ExpertStandard.dat  # map data
└── ExpertPlusStandard.dat  # map data
```

the .egg file is really just a .ogg audio file.
the .dat files are really just json:

### Info.dat
```json
{
  "_version": "2.0.0",
  "_songName": "Song Name",
  "_songSubName": "",
  "_songAuthorName": "Artist",
  "_levelAuthorName": "Mapper", 
  "_beatsPerMinute": 120,
  "_songTimeOffset": 0,
  "_shuffle": 0,
  "_shufflePeriod": 0.5,
  "_previewStartTime": 12,
  "_previewDuration": 10,
  "_songFilename": "song.egg",
  "_coverImageFilename": "cover.jpg",
  "_environmentName": "DefaultEnvironment",
  "_difficultyBeatmapSets": [...]
}
```

the only field we care about (for now) is `_beatsPerMinute` for aligning notes to audio frames.

### Map files (eg ExpertPlusStandard.dat)

### V3 Format (new)
```json
{
  "version": "3.3.0",
  "colorNotes": [
    {"b": 4.5, "x": 1, "y": 0, "c": 0, "d": 3, "a": 0}
  ]
}
```

### V2 Format (old)
```json
{
  "_version": "2.0.0",
  "_notes": [
    {"_time": 4.5, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 3}
  ]
}
```

there are other fields, but we can ignore them for the sake of this project.

### Field mappings:
- Beat/time: `b` (v3) or `_time` (v2) - beat number in song
- Position X: `x` (v3) or `_lineIndex` (v2) - lane 0-3 
- Position Y: `y` (v3) or `_lineLayer` (v2) - height 0-2
- Color: `c` (v3) or `_type` (v2) - 0=red, 1=blue, 2/3=bombs(skip)
- Direction: `d` (v3) or `_cutDirection` (v2) - 0-8 + 9(dot)

Special features: 
We are ignoring special v3 notes, BPM changes, noodle/mapping extensions, and offsets. Those will be excluded from the dataset before this sees it. 
Later, scrape/utils will need to handle that, but it can be ignored for now. I will update this spec when its needed.

## Using Encodec:

```python
encodec = create_audio_tokenizer(bandwidth=1.5)
tokens = encodec.encode(audio)
audio = encodec.decode(tokens)
```

encode takes [1, audio_samples] and returns [frames, n_codebooks]
decode takes [frames, n_codebooks] and returns [1, audio_samples]

when using bandwidth=1.5, encodec uses 2 codebooks.
each codebook is an integer from 0-1023.
for sequence modeling, they should be flattened:
[[c0_f0, c1_f0], [c0_f1, c1_f1], ...] -> [c0_f0, c1_f0, c0_f1, c1_f1, ...]
encodec encodes 24khz audio to 100 frames/second (240 samples/frame).

both the model and the input audio should be sent to cuda.

## Using Unsloth trainer:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-Base",
    max_seq_length=32768,
    load_in_4bit=True,
    full_finetuning=True,
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # no tokenizer passed! we are using pretokenized data
    data_collator=audio_masking_collator,
    args=TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1, # long sequence len needs small bs
        gradient_accumulation_steps=8, # simulate a higher bs
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=True,
        remove_unused_columns=False, # important for our custom tokenizing
        dataloader_drop_last=False,
    )
)

trainer.train()

# this only saves the tokenizer so inference doesnt yell at us trying to load the saved model
tokenizer.save_pretrained(args.output_dir)

trainer.save_model(args.output_dir)
```

full_finetuning=True instead of LoRA because this is a completely new task/token space, not teaching language modeling. the tokens will look like pure gibberish in language-space, but the transformer will do what transformers do best and figure out the data mapping. We are not building on *top* of qwen, we are reusing it.
audio_masking_collator sets labels to -100 for all audio tokens (and some special tokens), so model only learns to predict notes given audio, rather than wasting flops learning audio modeling.
only vocab indices specified above are used (our custom token space), rest of qwen's vocab is ignored. No changes should be made to the vocab length or anything, since the intention is to use off-the-shelf qwen and let it *learn* to only use our token space. Any editing of that will confuse it and make the code more complicated.