import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from tokenizers import Tokenizer
import subprocess
import imageio_ffmpeg
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import json
from datasets import load_from_disk
from tqdm import tqdm
import shutil
import time

# =====================================================================
# 2. MULTI-GPU & SLURM CONFIGURATION
# =====================================================================
# Fetch rank and world size from SLURM environment variables
# If not running under SLURM, defaults to 1 process (Single GPU mode)
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
RANK = int(os.environ.get("SLURM_PROCID", 0))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
time.sleep(LOCAL_RANK * 3)

# Dynamically assign the GPU based on local rank (e.g., cuda:0, cuda:1, cuda:2)
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

# 2026-07-23: see prototype/pipeline.py's identical change -- per-rank scratch
# moved off exa_project1 (over its project-wide inode soft limit, causing
# real silent data loss mid-run) onto exa_data1 (massive inode headroom).
SCRATCH_BASE = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/prototype_scratch"
os.makedirs(SCRATCH_BASE, exist_ok=True)
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
DTYPE = torch.float16 if "cuda" in DEVICE else torch.float32

# Logging strictly for Rank 0 to avoid terminal spam
def print_main(msg):
    if RANK == 0:
        print(msg)

print_main(f"🌍 Distributed Job Started: WORLD_SIZE={WORLD_SIZE}")
print(f"✅ Process {RANK} initialized on {DEVICE}")

FFMPEG_BIN = os.environ.get('FFMPEG_PATH')
DATASET_NAME = "HuggingFaceFV/finevideo"
MAX_VIDEOS = 9999999 # Set high enough to process the whole shard

# =====================================================================
# TOKENIZER CLASSES
# =====================================================================
# (These classes use the global DEVICE variable set above)

class Seed2Tokenizer:
    def __init__(self, target_size=512):
        self.folder = "./seed2"
        self.target_size = target_size
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self):        
        if os.path.exists(self.folder):
            sys.path.append(self.folder)
        try:
            from seed2_tokenizer import Seed2Tokenizer as LocalSeed2Tokenizer
            print(f"📦 [Rank {RANK}] [Seed2] Loading on {DEVICE}...")
            self.tokenizer = LocalSeed2Tokenizer.from_pretrained(
                self.folder, 
                torch_dtype=DTYPE
            ).to(DEVICE)
        except Exception as e:
            print(f"⚠️ [Rank {RANK}] [Seed2] Error loading: {e}")

    def encode_image(self, image_input):
        try:
            with torch.no_grad():
                tokens = self.tokenizer.encode_image(image_input) 
                if torch.is_tensor(tokens):
                    return tokens.flatten().detach().cpu().numpy().tolist()
                return tokens
        except Exception as e:
            return []

class CosmosVideoTokenizer:
    def __init__(self):
        self.model_name = "Cosmos-Tokenizer-DV8x16x16"
        self.tokenizer_path = f"pretrained_ckpts/{self.model_name}/encoder.jit"
        self.encoder = None
        self.load_tokenizer()

    def load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path): return
        print(f"📦 [Rank {RANK}] [Cosmos] Loading on {DEVICE}...")
        try:
            self.encoder = CausalVideoTokenizer(checkpoint_enc=self.tokenizer_path).to(DEVICE)
        except Exception as e:
            print(f"⚠️ [Rank {RANK}] [Cosmos] Error: {e}")

    def encode_video_chunk(self, frame_list, target_size=256):
        # 256 is Cosmos-Tokenizer-DV8x16x16's documented minimum supported
        # resolution (shorter side) -- the old target_size=160 default was
        # below spec. 2026-07-22 update: no crop at all now -- full-frame
        # aspect-preserving resize (shorter side = target_size, longer side
        # rounded to the nearest multiple of 16, the encoder's spatial
        # compression factor -- confirmed via direct test that the encoder
        # accepts non-square H/W as long as both are multiples of 16). The
        # previous version center-cropped to a square, which was carried over
        # from an earlier robot-camera pipeline (square-ish, heavily jittered
        # camera) and cropped real content off the sides of ordinary 16:9
        # YouTube video -- not appropriate now that this is omni-modal, not
        # robot-only. Token count is no longer size^2-based; it's
        # (H/16)*(W/16)*2, e.g. 256 shorter side on a 16:9 frame -> 256x448 ->
        # 16*28*2=896 tokens, vs 512 for a same-shorter-side square crop --
        # see REPORT.md #35 follow-up for the full before/after comparison.
        if self.encoder is None: return []
        try:
            w0, h0 = frame_list[0].size
            if w0 >= h0:
                new_h = target_size
                new_w = max(16, round(target_size * w0 / h0 / 16) * 16)
            else:
                new_w = target_size
                new_h = max(16, round(target_size * h0 / w0 / 16) * 16)

            processed_frames = []
            transform = T.Compose([
                T.Resize((new_h, new_w)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            for img in frame_list:
                tensor_frame = transform(img.convert("RGB"))
                processed_frames.append(tensor_frame)

            temporal_tensor = torch.stack(processed_frames)
            temporal_tensor = temporal_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE).to(DTYPE)

            with torch.no_grad():
                indices, _ = self.encoder.encode(temporal_tensor)
            return indices.flatten().detach().cpu().numpy().tolist()
        except Exception as e:
            print(f"❌ [Rank {RANK}] [Cosmos Error]: {e}")
            return []

class AVCLMTokenizer:
    def __init__(self, vocab_dir="avc_lm_v2"):
        self.vocab_dir = vocab_dir
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self):
        json_path = os.path.join(self.vocab_dir, "tokenizer.json")
        if not os.path.exists(json_path): return
        print(f"📦 [Rank {RANK}] [AVC-LM] Loading BPE from {self.vocab_dir}...")
        try:
            self.tokenizer = Tokenizer.from_file(json_path)
        except Exception as e:
            print(f"❌ [AVC-LM] Load error: {e}")

    def encode_h264_bitstream(self, h264_file_path):
        if self.tokenizer is None: return []
        try:
            with open(h264_file_path, "rb") as f:
                raw_bytes = f.read()
            text_data = raw_bytes.decode("latin-1")
            return self.tokenizer.encode(text_data).ids
        except Exception:
            return []

    def encode_mp4_segment(self, mp4_file_path, start_sec, duration_sec):
        if not os.path.exists(mp4_file_path): return []
        # Unique temp file per rank to avoid collisions
        temp_h264_path = f"temp_segment_rank{RANK}_{start_sec}.h264"
        try:
            command = [
                FFMPEG_BIN, "-y", "-ss", str(start_sec), "-i", mp4_file_path, 
                "-t", str(duration_sec), "-vf", "scale=256:256,fps=30", 
                "-c:v", "libx264", "-crf", "40", "-preset", "ultrafast", "-an", temp_h264_path
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            tokens = self.encode_h264_bitstream(temp_h264_path)
            if os.path.exists(temp_h264_path): os.remove(temp_h264_path)
            return tokens
        except subprocess.CalledProcessError:
            return []

# =====================================================================
# VLA DATA PIPELINE (Distributed Version)
# =====================================================================

class VLADatasetBuilder:
    def __init__(self, base_video_folder="./videos", base_jsonl_folder="./metadata", overlap_threshold=0.2):
        # 3. COLLISION PREVENTION: Assign unique working directories per rank
        self.video_folder = f"{base_video_folder}_rank_{RANK}"
        self.jsonl_folder = f"{base_jsonl_folder}_rank_{RANK}"
        self.temp_frames_dir = os.path.join(SCRATCH_BASE, f"temp_frames_rank_{RANK}")
        
        self.seed2 = Seed2Tokenizer()
        self.cosmos = CosmosVideoTokenizer()
        self.avc_lm = AVCLMTokenizer()
        self.target_fps = 30
        self.overlap_threshold = overlap_threshold
        
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.jsonl_folder, exist_ok=True)
        os.makedirs(self.temp_frames_dir, exist_ok=True)

    # --- Cleaning temp data

    def cleanup_temp_data(self):
        """
        Clears all contents inside the videos, metadata, and temp_frames directories
        to ensure no residual files consume disk space after processing a video.
        """
        folders_to_clean = [self.video_folder, self.jsonl_folder, self.temp_frames_dir]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path) # Remove standard files
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path) # Remove subdirectories if any
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to delete {file_path}. Reason: {e}")

    # --- METADATA PARSING METHODS ---

    @staticmethod
    def time_to_seconds(time_str):
        """Converts 'HH:MM:SS.mmm' to float seconds."""
        if not time_str: return 0.0
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0

    @staticmethod
    def calculate_overlap(start1, end1, start2, end2):
        """Calculates overlap duration in seconds between two intervals."""
        return max(0.0, min(end1, end2) - max(start1, start2))

    def parse_video_metadata(self, raw_data):
        """
        Converts raw FineVideo JSON into the design.jsonl hierarchical format.
        Applies multi-assignment logic for props, editing, and speech.
        """
        # Generate an ID or use the original filename
        video_id = raw_data.get("original_video_filename", "unknown").replace(".mp4", "")
        if video_id == "unknown":
            video_id = raw_data.get("youtube_title", "video").replace(" ", "_").lower()

        content_meta = raw_data.get("content_metadata", {})
        
        # Build the foundational target structure
        design_obj = {
            "video_id": video_id,
            "metadata": {
                "youtube_title": raw_data.get("youtube_title", ""),
                "category": raw_data.get("content_fine_category", ""),
                "resolution": raw_data.get("resolution", ""),
                "fps": content_meta.get("fps", 30.0),
                "duration_sec": raw_data.get("duration_seconds", 0)
            },
            "global_context": content_meta.get("description", ""),
            "scenes": []
        }

        raw_scenes = content_meta.get("scenes", [])
        timecoded_speech = raw_data.get("timecoded_text_to_speech", [])

        # Process Scenes
        for scene in raw_scenes:
            scene_start = self.time_to_seconds(scene.get("timestamps", {}).get("start_timestamp", ""))
            scene_end = self.time_to_seconds(scene.get("timestamps", {}).get("end_timestamp", ""))
            
            parsed_scene = {
                "scene_id": scene.get("sceneId", 0),
                "scene_title": scene.get("title", ""),
                "scene_time_range_sec": [round(scene_start, 3), round(scene_end, 3)],
                "scene_thematic": scene.get("thematicElements", ""),
                "scene_mood": scene.get("mood", {}).get("description", ""),
                "activities": []
            }

            # Process Activities
            for act_idx, activity in enumerate(scene.get("activities", [])):
                act_start = self.time_to_seconds(activity.get("timestamp", {}).get("start_timestamp", ""))
                act_end = self.time_to_seconds(activity.get("timestamp", {}).get("end_timestamp", ""))
                
                parsed_act = {
                    "activity_id": f"scene_{parsed_scene['scene_id']}_act_{act_idx + 1}",
                    "time_range_sec": [round(act_start, 3), round(act_end, 3)],
                    "text_prompt": activity.get("description", ""),
                    "speech_transcript": "", 
                    "props_present": [],
                    "video_editing": [],
                    "video_tokens": "" # Placeholder for the tokenizer pipeline
                }
                parsed_scene["activities"].append(parsed_act)

            # Map Props and Editing via Temporal Overlap
            for parsed_act in parsed_scene["activities"]:
                a_start, a_end = parsed_act["time_range_sec"]
                
                for prop in scene.get("props", []):
                    p_start = self.time_to_seconds(prop.get("timestamp", {}).get("start_timestamp", ""))
                    p_end = self.time_to_seconds(prop.get("timestamp", {}).get("end_timestamp", ""))
                    if self.calculate_overlap(a_start, a_end, p_start, p_end) >= self.overlap_threshold:
                        parsed_act["props_present"].append(prop.get("name", ""))
                
                for edit in scene.get("videoEditingDetails", []):
                    e_start = self.time_to_seconds(edit.get("timestamps", {}).get("start_timestamp", ""))
                    e_end = self.time_to_seconds(edit.get("timestamps", {}).get("end_timestamp", ""))
                    if self.calculate_overlap(a_start, a_end, e_start, e_end) >= self.overlap_threshold:
                        parsed_act["video_editing"].append(edit.get("description", ""))

            design_obj["scenes"].append(parsed_scene)

        # Map Speech globally across all scenes
        for speech in timecoded_speech:
            s_start = self.time_to_seconds(speech.get("start", ""))
            s_end = self.time_to_seconds(speech.get("end", ""))
            for parsed_scene in design_obj["scenes"]:
                for parsed_act in parsed_scene["activities"]:
                    a_start, a_end = parsed_act["time_range_sec"]
                    if self.calculate_overlap(s_start, s_end, a_start, a_end) >= self.overlap_threshold:
                        parsed_act["speech_transcript"] += speech.get("text", "") + " "

        # Cleanup trailing spaces
        for parsed_scene in design_obj["scenes"]:
            for parsed_act in parsed_scene["activities"]:
                parsed_act["speech_transcript"] = parsed_act["speech_transcript"].strip()

        return design_obj

    def extract_30fps_frames(self, video_path, start_sec, end_sec):
        duration = end_sec - start_sec
        # Use the rank-specific temp folder
        for f in os.listdir(self.temp_frames_dir): 
            os.remove(os.path.join(self.temp_frames_dir, f))

        try:
            command = [
                FFMPEG_BIN, "-y", "-ss", str(start_sec), "-i", video_path,
                "-t", str(duration), "-r", str(self.target_fps), "-f", "image2",
                os.path.join(self.temp_frames_dir, "frame_%04d.png")
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            frames = []
            frame_files = sorted([f for f in os.listdir(self.temp_frames_dir) if f.endswith(".png")])
            for file in frame_files:
                img_path = os.path.join(self.temp_frames_dir, file)
                img = Image.open(img_path).convert("RGB")
                frames.append(img.copy())
                img.close()
            return frames
        except Exception as e:
            return []

    def tokenize_activity_frames(self, frames, video_path, start_sec, duration):
        all_formatted_tokens = []
        seed2_token_count = 0
        cosmos_token_count = 0
        avclm_token_count = 0

        # 2026-07-22 (REPORT.md #38, "Option 2"): widened from 8 real frames
        # (0.267s @ 30fps) to 24 (0.8s) -- 8 frames was too short for the
        # model to see visible motion within a chunk (Huu's observation).
        # WINDOW_FRAMES is the real span every chunk covers (must match Phase
        # 3/4/5's --window-size/--window-frames for cosmos_N/agent_N to stay
        # aligned to the same real timespan). COSMOS_STRIDE samples every
        # Nth real frame from that span down to exactly 8 images for the
        # Cosmos encoder (which only ever wants 8 frames/chunk regardless of
        # resolution -- see REPORT.md #35's resolution ladder), so token
        # cost per chunk is unaffected by this change: WINDOW_FRAMES /
        # COSMOS_STRIDE must equal 8.
        WINDOW_FRAMES = 24
        COSMOS_STRIDE = 3
        assert WINDOW_FRAMES % COSMOS_STRIDE == 0 and WINDOW_FRAMES // COSMOS_STRIDE == 8, \
            "WINDOW_FRAMES/COSMOS_STRIDE must give exactly 8 samples for the Cosmos encoder"
        current_buffer = []

        # Iterate through every single frame to ensure perfect temporal alignment
        for idx, frame in enumerate(frames):
            # ------------------------------------------------------
            # 1. SEED2 (1 FPS): Triggers exactly at frame 0, 30, 60, etc.
            #    Unaffected by the window-size change -- runs on its own
            #    independent 1fps cadence, same as before.
            # ------------------------------------------------------
            if idx % self.target_fps == 0:
                temp_path = os.path.join(SCRATCH_BASE, f"temp_seed2_rank_{RANK}.jpg")
                frame.resize((self.seed2.target_size, self.seed2.target_size)).save(temp_path)
                seed2_ids = self.seed2.encode_image(temp_path)

                if seed2_ids:
                    all_formatted_tokens.append(f"<seed2> {' '.join(map(str, seed2_ids))} </seed2>")
                    seed2_token_count += len(seed2_ids)
                if os.path.exists(temp_path): os.remove(temp_path)

            # ------------------------------------------------------
            # 2. BUFFERING: Collect frames for Cosmos and AVC-LM
            # ------------------------------------------------------
            current_buffer.append(frame)

            # Once we hit WINDOW_FRAMES (24 real frames = 0.8s), process
            # Cosmos (stride-sampled down to 8) and AVC-LM (full window).
            if len(current_buffer) == WINDOW_FRAMES:
                # Calculate the start time for this specific window
                chunk_start_idx = idx - (WINDOW_FRAMES - 1)
                chunk_start_time = start_sec + (chunk_start_idx / self.target_fps)
                chunk_duration = WINDOW_FRAMES / self.target_fps

                # Encode Cosmos (Spatiotemporal) -- every COSMOS_STRIDE-th
                # real frame in the window, exactly 8 samples
                cosmos_frames = current_buffer[::COSMOS_STRIDE]
                cosmos_ids = self.cosmos.encode_video_chunk(cosmos_frames)
                if cosmos_ids:
                    all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, cosmos_ids))} </cosmos>")
                    cosmos_token_count += len(cosmos_ids)

                # Encode AVC-LM (Physical bitstream for this 0.8s segment --
                # widened to match, even though avc_lm output is stripped
                # before training, so chunk_timing stays self-consistent
                # across modalities within one activity)
                avc_ids = self.avc_lm.encode_mp4_segment(video_path, chunk_start_time, chunk_duration)
                if avc_ids:
                    all_formatted_tokens.append(f"<avc_lm> {' '.join(map(str, avc_ids))} </avc_lm>")
                    avclm_token_count += len(avc_ids)

                # Clear the buffer for the next chunk
                current_buffer = []

        # ------------------------------------------------------
        # 3. FINAL PADDING: Handle the remaining frames (e.g., last 6 frames)
        # ------------------------------------------------------
        if len(current_buffer) > 0:
            remaining_count = len(current_buffer)
            chunk_start_idx = len(frames) - remaining_count
            chunk_start_time = start_sec + (chunk_start_idx / self.target_fps)
            chunk_duration = remaining_count / self.target_fps

            # Pad the buffer to WINDOW_FRAMES for Cosmos architecture compatibility
            last_frame = current_buffer[-1]
            padding_needed = WINDOW_FRAMES - remaining_count
            current_buffer.extend([last_frame] * padding_needed)

            cosmos_frames = current_buffer[::COSMOS_STRIDE]
            cosmos_ids = self.cosmos.encode_video_chunk(cosmos_frames)
            if cosmos_ids:
                all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, cosmos_ids))} </cosmos>")
                cosmos_token_count += len(cosmos_ids)

            # Encode AVC-LM for the remaining actual duration
            avc_ids = self.avc_lm.encode_mp4_segment(video_path, chunk_start_time, chunk_duration)
            if avc_ids:
                all_formatted_tokens.append(f"<avc_lm> {' '.join(map(str, avc_ids))} </avc_lm>")
                avclm_token_count += len(avc_ids)

        return " ".join(all_formatted_tokens), seed2_token_count, cosmos_token_count, avclm_token_count
        
    def process_pipeline(self, output_base_name):
        # Each rank writes to its own JSONL file to prevent corruption
        output_jsonl = f"{output_base_name}_rank_{RANK}.jsonl"
        print(f"🚀 [Rank {RANK}] Started. Output: {output_jsonl}")

        # ==========================================================
        # FAULT TOLERANCE: GLOBAL RESUME MECHANISM
        # Scans ALL existing JSONL files from previous runs (any rank, CPU or GPU).
        # Ensures that if WORLD_SIZE changes, shards don't re-process duplicate videos.
        # ==========================================================
        import glob
        processed_video_ids = set()
        
        # Find all JSONL files from previous runs
        existing_files = glob.glob(f"{output_base_name}_rank_*.jsonl")
        
        if existing_files:
            print(f"🔄 [Rank {RANK}] Found {len(existing_files)} output files. Scanning for global resume...")
            for file_path in existing_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            try:
                                data = json.loads(line)
                                if "video_id" in data:
                                    processed_video_ids.add(data["video_id"])
                            except json.JSONDecodeError:
                                pass # Safely ignore corrupted lines from sudden crashes
                except Exception as e:
                    print(f"⚠️ [Rank {RANK}] Warning: Could not read {file_path}: {e}")
                    
            print(f"✅ [Rank {RANK}] Global Resume Ready: {len(processed_video_ids)} videos will be skipped.")

        try:
            dataset = load_from_disk(
                "/e/scratch/reformo/nguyen38/finevideo_disk"
            )
            
            # DATASET SHARDING
            if WORLD_SIZE > 1:
                dataset = dataset.shard(num_shards=WORLD_SIZE, index=RANK)
                print_main(f"✂️ Dataset successfully sharded into {WORLD_SIZE} parts.")
            
            # LẤY SỐ LƯỢNG CHÍNH XÁC CỦA SHARD NÀY
            shard_size = len(dataset)
            print(f"📊 [Rank {RANK}] This GPU's workload: {shard_size} videos")
            
            dataset_iter = iter(dataset)
        except Exception as e:
            print(f"❌ [Rank {RANK}] Failed to load dataset: {e}")
            return

        total_processed = 0
        pbar = tqdm(total=shard_size, desc=f"Rank {RANK} Progress", position=RANK)
        
        while total_processed < shard_size:
            try:
                item = next(dataset_iter)
                raw_metadata = item['json']
                
                # Extract video_id EARLY to check against our processed set
                video_id = raw_metadata.get("original_video_filename", "unknown").replace(".mp4", "")
                if video_id == "unknown":
                    video_id = raw_metadata.get("youtube_title", "video").replace(" ", "_").lower()

                # ==========================================================
                # SKIP LOGIC: If video_id is already in our set, fast-forward
                # ==========================================================
                if video_id in processed_video_ids:
                    total_processed += 1
                    pbar.update(1)
                    print(f'❌ [Rank {RANK}] skip video {video_id}')
                    continue # Skip heavy tokenization and video downloading completely

                # Proceed with downloading and processing ONLY for new videos
                video_bytes = item['mp4']
                design_obj = self.parse_video_metadata(raw_metadata)
                
                # Override video_id to ensure exact match with parse_video_metadata
                video_id = design_obj.get("video_id", "unknown")
                
                temp_video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)

                seed2_total_tokens = 0
                cosmos_total_tokens = 0
                avclm_total_tokens = 0
                video_total_duration = 0.0
                valid_activities_count = 0

                # Process activities...
                for scene in design_obj.get("scenes", []):
                    for activity in scene.get("activities", []):
                        time_range = activity.get("time_range_sec", [0.0, 0.0])
                        start_sec, end_sec = time_range[0], time_range[1]
                        duration = end_sec - start_sec

                        if duration <= 0:
                            activity["video_tokens"] = ""
                            continue

                        video_total_duration += duration
                        valid_activities_count += 1
                        
                        frames = self.extract_30fps_frames(temp_video_path, start_sec, end_sec)
                        if frames:
                            interleaved_tokens, seed2_token_count, cosmos_token_count, avclm_token_count = self.tokenize_activity_frames(frames, temp_video_path, start_sec, duration)
                            activity["video_tokens"] = interleaved_tokens
                            seed2_total_tokens += seed2_token_count
                            cosmos_total_tokens += cosmos_token_count
                            avclm_total_tokens += avclm_token_count
                        else:
                            activity["video_tokens"] = ""
                
                if valid_activities_count > 0 and video_total_duration > 0:
                    # 5. Print Stats
                    video_total_tokens = seed2_total_tokens + cosmos_total_tokens + avclm_total_tokens
                    seed2_per_sec = seed2_total_tokens / video_total_duration if video_total_duration > 0 else 0
                    cosmos_per_sec = cosmos_total_tokens / video_total_duration if video_total_duration > 0 else 0
                    avclm_per_sec = avclm_total_tokens / video_total_duration if video_total_duration > 0 else 0
                    token_per_sec = video_total_tokens / video_total_duration if video_total_duration > 0 else 0
                    print(f"🎬 [Rank {RANK}] Tokenized {total_processed + 1}/{shard_size} videos...")
                    print(f"📊 --- STATS: {video_id} ---")
                    print(f"✅ Total Duration  : {video_total_duration:.2f} seconds")
                    print(f"✅ Total Tokens    : {video_total_tokens:,} tokens")
                    print(f"✅ Seed2 Density         : {seed2_per_sec:.2f} tokens/second")
                    print(f"✅ Cosmos Density         : {cosmos_per_sec:.2f} tokens/second")
                    print(f"✅ Avc-lm Density         : {avclm_per_sec:.2f} tokens/second")
                    print(f"✅ Total Density         : {token_per_sec:.2f} tokens/second")

                    # Save locally
                    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(design_obj, ensure_ascii=False, separators=(',', ':')) + "\n")
                else:
                    print(f"⚠️ [Rank {RANK}] SKIPPING VIDEO: {video_id} (duration <= 0 or no activities)")
                    total_processed += 1
                    pbar.update(1)
                    continue
                
                total_processed += 1
                pbar.update(1)

            except StopIteration:
                print(f"[Rank {RANK}] End of dataset shard reached.")
                break
            except Exception as e:
                print(f"❌ [Rank {RANK}] Error: {e}")
                continue 
            finally:
                self.cleanup_temp_data()
                
        pbar.close()
        print(f"✨ [Rank {RANK}] Complete! Data saved to: {output_jsonl}")

if __name__ == "__main__":
    # Base folder names, will be dynamically appended with _rank_X in __init__
    builder = VLADatasetBuilder(
        base_video_folder=os.path.join(SCRATCH_BASE, "videos"),
        base_jsonl_folder=os.path.join(SCRATCH_BASE, "metadata"),
        overlap_threshold=0.2,
    )
    
    # We pass the base name, the class will output to e.g., training_ready_rank_0.jsonl
    # 2026-07-23: see prototype/pipeline.py's identical change -- renamed to
    # avoid the resume-skip logic silently no-op'ing on all already-processed
    # videos and never applying this session's cosmos/window fixes.
    # 2026-07-23: see prototype/pipeline.py's identical change -- output
    # moved off exa_project1 onto exa_data1.
    output_base_path = os.path.join(
        "/e/data1/datasets/playground/mmlaion/shared/nguyen38/window24_current/FineVideo-VLA",
        "training_ready_w24",
    )
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    builder.process_pipeline(output_base_name=output_base_path)