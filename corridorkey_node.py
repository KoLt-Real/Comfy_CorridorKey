import os
import sys
import torch
import urllib.request
import numpy as np
import cv2
import tempfile
import shutil

node_dir = os.path.dirname(os.path.abspath(__file__))
if node_dir not in sys.path:
    sys.path.append(node_dir)

from CorridorKeyModule.inference_engine import CorridorKeyEngine

class CorridorKeyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "alpha_hint": ("MASK",),
                "alpha_generator": (["None", "GVM (Auto)", "VideoMaMa (from mask)"], {"default": "None"}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                "use_refiner": ("BOOLEAN", {"default": True}),
                "input_is_linear": ("BOOLEAN", {"default": False}),
                "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "auto_despeckle": ("BOOLEAN", {"default": True}),
                "despeckle_size": ("INT", {"default": 400, "min": 0, "max": 10000}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "composite")
    FUNCTION = "process"
    CATEGORY = "CorridorKey"

    def process(self, images, alpha_hint=None, alpha_generator="None", device="cuda", 
                use_refiner=True, input_is_linear=False, despill_strength=1.0, 
                auto_despeckle=True, despeckle_size=400, verbose=True):
        
        # Safe ComfyUI progress bar import
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(images.shape[0])
        except ImportError:
            pass
        
        B, H, W, C = images.shape
        
        # 1. Handle Alpha Generation
        final_masks_to_use = []
        
        if alpha_generator == "GVM (Auto)":
            if verbose: print("[CorridorKey] Generating alpha hint with GVM...")
            temp_in = tempfile.mkdtemp()
            temp_out = tempfile.mkdtemp()
            try:
                for i in range(B):
                    img_np = (images[i].cpu().numpy() * 255.0).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(temp_in, f"{i:05d}.png"), img_bgr)
                
                # Import GVM
                gvm_dir = os.path.join(node_dir, "gvm_core")
                if gvm_dir not in sys.path: sys.path.append(gvm_dir)
                try:
                    from gvm_core.wrapper import GVMProcessor
                except ImportError as e:
                    raise ImportError(f"[CorridorKey] Failed to import GVM. Ensure you ran 'uv sync' or installed the requirements from gvm_core. Error: {e}")
                
                # Auto-download GVM weights if missing
                gvm_weights = os.path.join(gvm_dir, "weights")
                os.makedirs(gvm_weights, exist_ok=True)
                if not os.path.exists(os.path.join(gvm_weights, "model_index.json")):
                    if verbose: print("[CorridorKey] GVM Model (80GB+) not found locally. Auto-downloading from HuggingFace... (This may take a very long time)")
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id="geyongtao/gvm", local_dir=gvm_weights)
                    except Exception as e:
                        print(f"[CorridorKey Error] GVM weights download failed: {e}")
                
                # Ensure model paths are relative to gvm_core instead of running dir
                processor = GVMProcessor(model_base=gvm_weights, device=device)
                
                processor.process_sequence(
                    input_path=temp_in,
                    output_dir=None,
                    num_frames_per_batch=1,
                    decode_chunk_size=1,
                    denoise_steps=1,
                    mode="matte",
                    write_video=False,
                    direct_output_dir=temp_out
                )
                
                # Read masks back
                out_files = sorted([f for f in os.listdir(temp_out) if f.endswith(".png")])
                for f in out_files:
                    m = cv2.imread(os.path.join(temp_out, f), cv2.IMREAD_GRAYSCALE)
                    final_masks_to_use.append(torch.from_numpy(m).float() / 255.0)
                
                del processor
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            finally:
                shutil.rmtree(temp_in, ignore_errors=True)
                shutil.rmtree(temp_out, ignore_errors=True)
                
        elif alpha_generator == "VideoMaMa (from mask)":
            if alpha_hint is None:
                raise ValueError("[CorridorKey Error] Using 'VideoMaMa' requires an 'alpha_hint' (mask) connected. Please draw a rough mask or use another node to generate a coarse hint first.")
            if verbose: print("[CorridorKey] Refining alpha hint with VideoMaMa...")
            
            B_mask = alpha_hint.shape[0] if len(alpha_hint.shape) == 3 else 1
            if len(alpha_hint.shape) == 2:
                alpha_hint_unsqueeze = alpha_hint.unsqueeze(0)
            else:
                alpha_hint_unsqueeze = alpha_hint
                
            input_frames = []
            mask_frames = []
            for i in range(B):
                img_np = (images[i].cpu().numpy() * 255.0).astype(np.uint8)
                input_frames.append(img_np)
                
                m_idx = min(i, B_mask - 1)
                m_np = (alpha_hint_unsqueeze[m_idx].cpu().numpy() * 255.0).astype(np.uint8)
                mask_frames.append(m_np)
            
            vmm_path = os.path.join(node_dir, "VideoMaMaInferenceModule")
            if vmm_path not in sys.path: sys.path.append(vmm_path)
            try:
                from VideoMaMaInferenceModule.inference import load_videomama_model, run_inference
            except ImportError as e:
                raise ImportError(f"[CorridorKey] Failed to import VideoMaMa. Ensure you ran 'uv sync' or installed the requirements from VideoMaMaInferenceModule. Error: {e}")
            
            vmm_chk_path = os.path.join(vmm_path, "checkpoints")
            os.makedirs(vmm_chk_path, exist_ok=True)
            if not os.path.exists(os.path.join(vmm_chk_path, "VideoMaMa", "diffusion_pytorch_model.safetensors")):
                if verbose: print("[CorridorKey] VideoMaMa Model not found locally. Auto-downloading from HuggingFace...")
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id="SammyLim/VideoMaMa", local_dir=vmm_chk_path)
                except Exception as e:
                    print(f"[CorridorKey Error] VideoMaMa weights download failed: {e}")
            
            base_m = os.path.join(vmm_chk_path, "stable-video-diffusion-img2vid-xt")
            unet_m = os.path.join(vmm_chk_path, "VideoMaMa")
            
            pipeline = load_videomama_model(base_model_path=base_m, unet_checkpoint_path=unet_m, device=device)
            gen = run_inference(pipeline, input_frames, mask_frames, chunk_size=24)
            
            for chunk in gen:
                for frame in chunk:
                    m_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    final_masks_to_use.append(torch.from_numpy(m_gray).float() / 255.0)
                    
            del pipeline
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        else: # "None"
            if alpha_hint is None:
                raise ValueError("[CorridorKey Error] 'alpha_hint' must be connected! If you don't have a hint mask, use 'alpha_generator -> GVM (Auto)'.")
            
            B_mask = alpha_hint.shape[0] if len(alpha_hint.shape) == 3 else 1
            if len(alpha_hint.shape) == 2:
                alpha_hint_unsqueeze = alpha_hint.unsqueeze(0)
            else:
                alpha_hint_unsqueeze = alpha_hint
                
            for i in range(B):
                m_idx = min(i, B_mask - 1)
                final_masks_to_use.append(alpha_hint_unsqueeze[m_idx])
                
        # 2. Check for missing CorridorKey Model
        checkpoint_dir = os.path.join(node_dir, "CorridorKeyModule", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "CorridorKey.pth")

        if not os.path.exists(checkpoint_path):
            if verbose: print("[CorridorKey] Model not found locally. Downloading from HuggingFace...")
            model_url = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
            try:
                torch.hub.download_url_to_file(model_url, checkpoint_path, progress=True)
            except AttributeError:
                urllib.request.urlretrieve(model_url, checkpoint_path)
            if verbose: print("[CorridorKey] Download complete.")

        # 3. Initialize CorridorKey Engine
        if verbose: print("[CorridorKey] Initializing Engine...")
        engine = CorridorKeyEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=2048,
            use_refiner=use_refiner
        )

        output_masks = []
        output_images = []
        
        # 4. Process each frame
        if verbose: print(f"[CorridorKey] Processing {B} frames...")
        for i in range(B):
            if verbose: print(f"  -> Inferencing final keying: Frame {i+1}/{B}")
            img = images[i].cpu().numpy()  # [H, W, 3]
            
            mask_idx = min(i, len(final_masks_to_use) - 1)
            mask = final_masks_to_use[mask_idx].cpu().numpy()  # [H, W]

            result = engine.process_frame(
                image=img,
                mask_linear=mask,
                input_is_linear=input_is_linear,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size
            )

            out_alpha = result["alpha"]  # [H, W] or [H, W, 1]
            if len(out_alpha.shape) == 3:
                out_alpha = out_alpha[:, :, 0]
            output_masks.append(torch.from_numpy(out_alpha))
            
            # The 'comp' output is the keyed subject over a dark checkerboard
            out_comp = result["comp"]
            output_images.append(torch.from_numpy(out_comp))
            
            if pbar is not None:
                pbar.update(1)

        final_masks = torch.stack(output_masks)
        final_images = torch.stack(output_images)
        
        # Free memory to ensure ComfyUI doesn't OOM between runs
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if verbose: print("[CorridorKey] Processing Complete.")
        return (final_masks, final_images)

