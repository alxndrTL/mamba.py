# ------------------------------------------------------------
# 0️⃣  Imports & tiny helpers (tokenizer / detokenizer)
# ------------------------------------------------------------
import json, pathlib, os
from collections import deque
import time
from typing import List
# from mamba_lm_mlx import MambaLM, MambaLMConfig
import mlx.core as mx
import numpy 
from mlx_lm import load, generate
# ------------------------------------------------------------
# 1️⃣  Dummy tokenizer (replace with your real tokenizer)
# ------------------------------------------------------------
def tokenize(s: str) -> List[int]:
    """Very naive whitespace → list of ints.
    Replace with a real tokenizer that returns token ids."""
    # Using Unicode code‑point as a cheap stand‑in.
    return [ord(ch) for ch in s if not ch.isspace()]

def detokenize(ids: List[int]) -> str:
    """Very naive reverse of `tokenize`."""
    return "".join(chr(i) for i in ids)

# ------------------------------------------------------------
# 2️⃣  Minimal model wrapper -------------------------------------------------
# ------------------------------------------------------------
# NOTE: The script expects a **MLX‑compatible model** that can be loaded
# with `mamba.load("<path>")`.  If you do not have such a checkpoint yet,
# you can still run the file – it will skip model generation and just
# print a placeholder.  (See the “Running on an M4” section below for
# details on getting a real model.)
try:
    
    # import mamba  # <-- pip install mamba-llm (or use your own fork)
    _model = load.load("NVIDIA-Nemotron-3-Nano30B-A3B-MLX-8bit")  # <-- CHANGE ME
    _model = _model.to("mlx")                     # move to MLX device
except Exception as exc:
    raise RuntimeError(
        "Make sure `mamba` is installed and POINT `_model` to a valid MLX checkpoint."
    ) from exc

def generate_one(token_id: int, prefix_embedding: mx.array = None):
    """
    Very thin wrapper around the underlying model.
    It returns a tuple ``(next_token_id, token_embedding)``.
    If your model only returns an id, you can drop the embedding part
    and keep only the id.
    """
    # The real Mamba API returns a `GenerationOutput` object with many fields.
    # For the demo we assume it exposes ``next_token`` and ``embedding``.
    out = _model.generate_one(token_id, prefix_embedding=prefix_embedding)
    next_id = int(out.token)                     # token id
    embed   = out.embedding                       # (feature_dim,)
    return next_id, embed

# ------------------------------------------------------------
# 3️⃣  Short‑Term Memory (with EMA & goal placeholders)
# ------------------------------------------------------------
class ShortTermMemory:
    def __init__(self,
                 feature_dim: int = 256,
                 max_history: int = 32,
                 ema_alpha: float = 0.1,
                 age_beta: float = 0.001,
                 goal_dim: int = 256):
        self.feature_dim   = feature_dim
        self.max_history   = max_history
        self.ema_alpha     = ema_alpha
        self.age_beta      = age_beta
        self.time_step     = 0
        self.buffer      = deque(maxlen=max_history)   # stores fused vecs
        self.weights     = []                             # parallel list
        self.timestamps  = []                             # parallel list
        self.goal_vecs   = [mx.zeros(goal_dim, dtype=mx.float32)
                            for _ in range(max_history)]

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a, b):
        return float(mx.dot(a / mx.norm(a), b / mx.norm(b)))

    # ------------------------------------------------------------------
    def _find_nearest(self, query):
        if not self.buffer:
            return None, 0.0
        sims = [self._cosine_similarity(query, mem) for mem in self.buffer]
        best_i = int(mx.argmax(mx.array(sims)).item())
        return best_i, sims[best_i]

    # ------------------------------------------------------------------
    def add(self, audio: mx.array, text: mx.array, energy: float = 1.0):
        """
        Fuse audio+text, store in STM.
        If a similar slot exists, blend via EMA; otherwise append a fresh slot.
        """
        fused = mx.concatenate([audio.astype(mx.float32),
                                text.astype(mx.float32)], axis=0)

        idx, sim = self._find_nearest(fused)

        if idx is not None and sim > 0.6:                # lower thresh → more updates
            # ----- EMA blend the vector -----
            old_vec = self.buffer[idx]
            new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * fused
            self.buffer[idx] = new_vec

            # ----- EMA blend the scalar energy -----
            old_w   = self.weights[idx]
            new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.weights[idx] = new_w

            # ----- Refresh timestamp (age matters for later decay) -----
            self.timestamps[idx] = self.time_step
            # store the goal vector (currently just zeros – you can replace it)
            self.goal_vecs[idx] = self._current_global_goal()
        else:
            # ----- Append a brand‑new slot -----
            self.buffer.append(fused)
            self.weights.append(energy)
            self.timestamps.append(self.time_step)
            self.goal_vecs.append(self._current_global_goal())

            # Trim auxiliary lists if we exceeded the ring size
            if len(self.weights) > self.max_history:
                self.weights.pop(0)
                self.timestamps.pop(0)
                self.goal_vecs.pop(0)

        self.time_step += 1

    # ------------------------------------------------------------------
    def _current_global_goal(self):
        # In a full system this would read from a shared GoalUpdater.
        # For the demo we just return zeros (no bias yet).
        return mx.zeros(self.feature_dim, dtype=mx.float32)

    # ------------------------------------------------------------------
    #  📦  Persistence (JSON) -------------------------------------------------
    def to_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "max_history": self.max_history,
            "ema_alpha": self.ema_alpha,
            "age_beta": self.age_beta,
            "time_step": self.time_step,
            "buffer": [v.tolist() for v in self.buffer],
            "weights": list(self.weights),
            "timestamps": list(self.timestamps),
            "goal_vecs": [g.tolist() for g in self.goal_vecs],
        }

    @classmethod
    def from_dict(cls, payload: dict):
        stm = cls(
            feature_dim=payload["feature_dim"],
            max_history=payload["max_history"],
            ema_alpha=payload["ema_alpha"],
            age_beta=payload["age_beta"],
            goal_dim=payload.get("goal_dim", 256),
        )
        stm.time_step = payload["time_step"]
        stm.buffer = [mx.array(v) for v in payload["buffer"]]
        stm.weights = payload["weights"]
        stm.timestamps = payload["timestamps"]
        stm.goal_vecs = [mx.array(v) for v in payload["goal_vecs"]]
        return stm

    def save(self, path: str):
        data = self.to_dict()
        pathlib.Path(path).write_text(json.dumps(data, indent=2))
        print(f"[STM] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        payload = json.loads(pathlib.Path(path).read_text())
        return cls.from_dict(payload)

# ------------------------------------------------------------
# 4️⃣  Long‑Term Memory (codebook)
# ------------------------------------------------------------
class LongTermMemory:
    def __init__(self,
                 feature_dim: int = 256,
                 max_entries: int = 128,
                 ema_alpha: float = 0.2,
                 novelty_thresh: float = 0.6,
                 min_age_before_prune: int = 30):
        self.feature_dim = feature_dim
        self.max_entries = max_entries
        self.ema_alpha   = ema_alpha
        self.novelty_thresh = novelty_thresh
        self.min_age_before_prune = min_age_before_prune
        self.entries = []                     # (vec, weight, timestamp)

    @staticmethod
    def _cosine_similarity(a, b):
        return float(mx.dot(a / mx.norm(a), b / mx.norm(b)))

    def _nearest(self, query):
        if not self.entries:
            return None, 0.0
        sims = [self._cosine_similarity(query, vec) for vec, _, _ in self.entries]
        best_i = int(mx.argmax(mx.array(sims)).item())
        return best_i, sims[best_i]

    def add(self, vec: mx.array, energy: float = 1.0):
        idx, sim = self._nearest(vec)
        if idx is not None and sim > self.novelty_thresh:
            old_vec, old_w, _ = self.entries[idx]
            new_vec = (1.0 - self.ema_alpha) * old_vec + self.ema_alpha * vec
            new_w   = (1.0 - self.ema_alpha) * old_w + self.ema_alpha * energy
            self.entries[idx] = (new_vec, new_w, self.time_step if hasattr(self, "time_step") else 0)
        else:
            if len(self.entries) < self.max_entries:
                self.entries.append((vec, energy, 0))
            else:
                # prune the *oldest* low‑weight entry
                min_idx = int(mx.argmin(mx.array([w for _, w, _ in self.entries])).item())
                self.entries[min_idx] = (vec, energy, 0)

    def recall(self, query, top_k=5):
        sims = [(i, self._cosine_similarity(query, vec))
                for i, (vec, _, _) in enumerate(self.entries)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def prune(self, current_time: int):
        self.entries = [(v, w, t) for v, w, t in self.entries
                        if current_time - t < self.min_age_before_prune]

    # ------------------------------------------------------------------
    #  📦  Persistence (JSON) -------------------------------------------------
    def to_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "max_entries": self.max_entries,
            "ema_alpha": self.ema_alpha,
            "novelty_thresh": self.novelty_thresh,
            "min_age_before_prune": self.min_age_before_prune,
            "entries": [
                {"vec": v.tolist(),
                 "weight": w,
                 "timestamp": t}
                for v, w, t in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict):
        lt = cls(
            feature_dim=payload["feature_dim"],
            max_entries=payload["max_entries"],
            ema_alpha=payload["ema_alpha"],
            novelty_thresh=payload["novelty_thresh"],
            min_age_before_prune=payload["min_age_before_prune"],
        )
        lt.entries = [(mx.array(e["vec"]), e["weight"], e["timestamp"])
                      for e in payload["entries"]]
        return lt

    def save(self, path: str):
        payload = self.to_dict()
        pathlib.Path(path).write_text(json.dumps(payload, indent=2))
        print(f"[LTM] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        payload = json.loads(pathlib.Path(path).read_text())
        return cls.from_dict(payload)

# ------------------------------------------------------------
# 5️⃣  Goal Updater (EMA‑driven intention)
# ------------------------------------------------------------
class GoalUpdater:
    """
    Holds a single intention vector that is updated by EMA.
    The vector can be queried (`current`) or used as a conditioning signal.
    """
    def __init__(self, dim: int = 256, ema_alpha: float = 0.05):
        self.current = mx.zeros(dim, dtype=mx.float32)
        self.ema_alpha = ema_alpha

    def update(self, reference: mx.array):
        """EMA‑blend `reference` into the stored goal."""
        self.current = (1.0 - self.ema_alpha) * self.current + self.ema_alpha * reference

    def as_embedding(self) -> mx.array:
        """Return the goal as a (1, dim) tensor ready for concatenation."""
        return self.current / mx.norm(self.current, keepdims=True)[None, :]

# ------------------------------------------------------------
# 6️⃣  Memory Observer (consolidation, pruning, counter‑factual)
# ------------------------------------------------------------
class MemoryObserver:
    def __init__(self,
                 short: ShortTermMemory,
                 long: LongTermMemory,
                 consolidate_thresh: float = 0.8,
                 novelty_check_every: int = 50,
                 age_beta: float = 0.001):
        self.short          = short
        self.long           = long
        self.consolidate_thresh = consolidate_thresh
        self.novelty_check_every = novelty_check_every
        self.age_beta       = age_beta
        self.time_since_last = 0

        # goal updater will be attached later by the engine
        self._goal_updater = None

        # keep a reference to the *global* goal updater that lives in the engine
        self._goal_updater = None

    # ------------------------------------------------------------------
    def maybe_consolidate(self):
        """Consolidate the most salient STM slot into LTM."""
        if self.short.time_step == 0:
            return

        ages = mx.array([self.short.time_step - t for t in self.short.timestamps])
        priorities = mx.array(self.short.weights) * mx.exp(-self.age_beta * ages)
        best_idx = int(mx.argmax(priorities).item())
        best_weight = self.short.weights[best_idx]

        if best_weight >= self.consolidate_thresh:
            vec_to_store = self.short.buffer[best_idx]
            self.long.add(vec_to_store, energy=best_weight)

            # update the *global* goal via EMA (the engine holds the updater)
            if self._goal_updater is not None:
                self._goal_updater.update(vec_to_store)

    # ------------------------------------------------------------------
    def prune_long_term(self):
        """Drop entries that are older than `min_age_before_prune`."""
        self.long.prune(self.short.time_step)

    # ------------------------------------------------------------------
    def counterfactual_score(self,
                             query_vec: mx.array,
                             sigma: float = 0.1) -> float:
        """
        Perturb `query_vec` with small Gaussian noise and return the highest
        cosine similarity to any LTM entry.  Higher = “I’m close to something I already know”.
        """
        noise = mx.random.normal(shape=query_vec.shape, std=sigma)
        cand = query_vec + noise
        cand = cand / mx.norm(cand)
        query = query_vec / mx.norm(query_vec)

        sims = [mx.dot(cand, entry[0]) for entry in self.long.entries]
        return float(mx.max(mx.array(sims))) if sims else 0.0

    # ------------------------------------------------------------------
    #  📦  Persistence (JSON + NumPy) ---------------------------------------
    def save_all(self, folder: str):
        """Dump STM, LTM and the current global goal."""
        os.makedirs(folder, exist_ok=True)

        # STM
        stm_path = os.path.join(folder, "short_term.json")
        self.short.save(stm_path)

        # LTM
        lt_path = os.path.join(folder, "long_term.json")
        self.long.save(lt_path)

        # Global goal (as a .npy file – easy to reload)
        goal_path = os.path.join(folder, "global_goal.npy")
        mx.save_numpy(goal_path, self._goal_updater.current if self._goal_updater else mx.zeros_like(self.long.entries[0][0]) if self.long.entries else mx.zeros(self.short.feature_dim, dtype=mx.float32))
        print(f"[Observer] Full snapshot written to {folder}")

    @classmethod
    def load_all(cls, folder: str):
        """Re‑create a MemoryObserver from a previously saved snapshot."""
        stm_path = os.path.join(folder, "short_term.json")
        lt_path  = os.path.join(folder, "long_term.json")
        goal_path = os.path.join(folder, "global_goal.npy")

        short = ShortTermMemory.load(stm_path)
        long  = LongTermMemory.load(lt_path)

        # Load goal vector
        goal_vec = mx.load(goal_path)

        # Build a fresh observer and inject the goal into its updater
        obs = cls(short=short, long=long,
                  consolidate_thresh=0.8,
                  novelty_check_every=50,
                  age_beta=0.001)
        # create a dummy GoalUpdater and set its `.current` to the loaded vector
        updater = GoalUpdater(dim=goal_vec.shape[0], ema_alpha=0.03)
        updater.current = goal_vec
        obs._goal_updater = updater
        return obs

# ------------------------------------------------------------
# 7️⃣  Dual‑Buffer Engine (ties everything together)
# ------------------------------------------------------------
class DualMemoryEngine:
    """
    High‑level façade that:
    * tokenises a prompt,
    * generates token‑by‑token,
    * stores embeddings in STM,
    * consolidates into LTM,
    * updates the EMA‑goal,
    * (optionally) injects the goal into the model,
    * can be persisted to JSON.
    """
    def __init__(self,
                 short_cfg: dict,
                 long_cfg: dict,
                 goal_dim: int = 256):
        self.short = ShortTermMemory(**short_cfg, goal_dim=goal_dim)
        self.long  = LongTermMemory(**long_cfg)

        self.observer = MemoryObserver(
            short=self.short,
            long=self.long,
            consolidate_thresh=0.8,
            novelty_check_every=50,
            age_beta=0.001,
        )
        # give the observer a handle to the GoalUpdater that lives in the engine
        self.observer._goal_updater = self._goal_updater
        self._goal_updater = GoalUpdater(dim=goal_dim, ema_alpha=0.03)

    # ------------------------------------------------------------------
    def _goal_embedding(self) -> mx.array:
        """Return the current global goal as a (1, dim) tensor."""
        return self._goal_updater.as_embedding()

    # ------------------------------------------------------------------
    def generate_and_learn(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Core generation loop.
        * Tokenises `prompt`.
        * For each generated token:
            - asks the model for the next id **and** its embedding,
            - builds a fused audio+text embedding,
            - stores it in STM,
            - periodically consolidates into LTM,
            - updates the EMA‑goal,
            - optionally uses the goal embedding as conditioning.
        """
        token_ids = tokenize(prompt)
        generated: List[int] = []

        # dummy audio embedding – replace with real audio features later
        dummy_audio_dim = self.short.feature_dim // 2
        dummy_audio = mx.random.normal((dummy_audio_dim,), dtype=mx.float32)

        # static goal prefix that we will prepend to *every* token embedding
        goal_prefix = self._goal_embedding()          # shape (1, dim)

        for tid in token_ids:
            # -------------------- generation step --------------------
            next_id, token_emb = generate_one(
                token_id=tid,
                prefix_embedding=goal_prefix          # <<< inject goal
            )
            generated.append(next_id)

            # -------------------- build embeddings --------------------
            txt_emb = token_emb                       # (feature_dim,)
            audio_emb = dummy_audio                  # (dummy_audio_dim,)

            # -------------------- store in STM --------------------
            reward = 1.0 if 32 <= next_id < 127 else 0.0
            self.short.add(audio_emb, txt_emb, energy=reward)

            # -------------------- periodic consolidation ------------
            if self.short.time_step % 5 == 0:
                latest_vec = self.short.buffer[-1]    # newest fused vector
                self.long.add(latest_vec, energy=reward)

                # update the EMA‑goal with the same vector
                self._goal_updater.update(latest_vec)

                # optional counter‑factual novelty signal
                if self.short.time_step % self.observer.novelty_check_every == 0:
                    cf_score = self.observer.counterfactual_score(latest_vec)
                    print(f"[CF] Novelty score: {cf_score:.3f}")

        # -------------------------------------------------------------
        # final housekeeping
        # -------------------------------------------------------------
        self.observer.prune_long_term()
        self.observer.maybe_consolidate()

        return detokenize(generated)

    # ------------------------------------------------------------------
    #  📂  Expose a convenience method to dump the whole memory pack
    def dump_snapshot(self, folder: str):
        self.observer.save_all(folder)

    @classmethod
    def load_snapshot(cls, folder: str):
        """Factory that builds an engine and restores its memory."""
        restored = cls(short_cfg={"feature_dim": 256, "max_history": 32},
                       long_cfg={"feature_dim": 256, "max_entries": 128},
                       goal_dim=256)
        # replace the internal memory objects with the restored ones
        restored.short = MemoryObserver.load_all(folder).short
        restored.long  = MemoryObserver.load_all(folder).long
        restored._goal_updater = GoalUpdater(dim=256, ema_alpha=0.03)
        restored._goal_updater.current = mx.load(
            os.path.join(folder, "global_goal.npy")
        )
        # re‑attach the observer (it holds a reference to the updater)
        restored.observer = MemoryObserver(
            short=restored.short,
            long=restored.long,
            consolidate_thresh=0.8,
            novelty_check_every=50,
            age_beta=0.001,
        )
        restored.observer._goal_updater = restored._goal_updater
        return restored

# ------------------------------------------------------------
# 8️⃣  Tiny demo when the file is executed directly
# ------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # 8.1  Build the engine (feel free to tweak the configs)
    # ------------------------------------------------------------
    engine = DualMemoryEngine(
        short_cfg={"feature_dim": 256, "max_history": 32},
        long_cfg = {"feature_dim": 256, "max_entries": 128},
        goal_dim   = 256,
    )

    # ------------------------------------------------------------
    # 8.2  Prompt
    # ------------------------------------------------------------
    prompt = "Tell me a short story about a robot that learns to paint."
    print("\n=== Prompt ===")
    print(prompt)

    # ------------------------------------------------------------
    # 8.3  Generate + learn
    # ------------------------------------------------------------
    story = engine.generate_and_learn(prompt, temperature=1.0)
    print("\n=== Generated story ===")
    print(story)

    # ------------------------------------------------------------
    # 8.4  Inspect LTM matches for the newest fused embedding
    # ------------------------------------------------------------
    query_vec = engine.short.buffer[-1]          # newest fused vector
    idxs, scores = engine.long.recall(query_vec, top_k=5)
    print("\nTop LTM matches (index, similarity):")
    for i, s in zip(idxs, scores):
        print(f"  {i:02d} → {s:.3f}")

    # ------------------------------------------------------------
    # 8.5  Look at the EMA‑updated global goal
    # ------------------------------------------------------------
    print("\nCurrent global goal (first 5 dims):",
          engine._goal_updater.current[:5].tolist())

    # ------------------------------------------------------------
    # 8.6  Persist everything to disk
    # ------------------------------------------------------------
    snapshot_dir = "memory_snapshot_demo"
    engine.dump_snapshot(snapshot_dir)
    print(f"\n[Demo] Snapshot written to ./{snapshot_dir}")

    # ------------------------------------------------------------
    # 8.7  (Optional) Load the snapshot back and continue generation
    # ------------------------------------------------------------
    # restored = DualMemoryEngine.load_snapshot(snapshot_dir)
    # print("\n--- Continued after reload ---")
    # restored.generate_and_learn("And then the robot ...", temperature=1.0)
