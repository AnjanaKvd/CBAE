import numpy as np
import io
import h5py

from .constants import (
    N_SLOTS,
    N_CTRL_PTS,
    SLOT_BG_STATIC,
    SLOT_BG_DYNAMIC,
    SLOT_BODY,
    SLOT_FACE,
    SLOT_MOUTH,
    SLOT_SECONDARY,
    SLOT_DYNAMIC,
)


class CRFTensor:
    def __init__(self, n_slots=N_SLOTS, n_ctrl_pts=N_CTRL_PTS):
        self.P = np.zeros((n_slots, n_ctrl_pts, 2), dtype=np.float16)
        self.c = np.zeros((n_slots, 3), dtype=np.float16)
        self.alpha = np.zeros(n_slots, dtype=np.float16)
        self.alive = np.full(n_slots, -5.0, dtype=np.float16)  # sigmoid(-5) ≈ 0.007
        self.csg = np.zeros(n_slots, dtype=bool)
        self.z = np.zeros(n_slots, dtype=np.int8)

    def active_slots(self, threshold=0.1):
        """Returns array of active slot indices where sigmoid(alive) > threshold"""
        alive_f32 = self.alive.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-alive_f32))
        return np.where(sig > threshold)[0]

    def slot_block(self, index):
        """Returns string label for which semantic block a slot belongs to."""
        if SLOT_BG_STATIC[0] <= index <= SLOT_BG_STATIC[1]:
            return "background_static"
        if SLOT_BG_DYNAMIC[0] <= index <= SLOT_BG_DYNAMIC[1]:
            return "background_dynamic"
        if SLOT_BODY[0] <= index <= SLOT_BODY[1]:
            return "body"
        if SLOT_FACE[0] <= index <= SLOT_FACE[1]:
            return "face"
        if SLOT_MOUTH[0] <= index <= SLOT_MOUTH[1]:
            return "mouth"
        if SLOT_SECONDARY[0] <= index <= SLOT_SECONDARY[1]:
            return "secondary"
        if SLOT_DYNAMIC[0] <= index <= SLOT_DYNAMIC[1]:
            return "dynamic"
        return "unknown"

    def set_shape(self, slot_idx, P, c, alpha, csg=False):
        """Convenience setter with value clamping."""
        self.P[slot_idx] = np.clip(P, 0.0, 1.0).astype(np.float16)
        self.c[slot_idx] = np.clip(c, 0.0, 1.0).astype(np.float16)
        self.alpha[slot_idx] = np.clip(alpha, 0.0, 1.0).astype(np.float16)
        self.csg[slot_idx] = bool(csg)

    def activate(self, slot_idx):
        """Sets alive to +5.0 (sigmoid ≈ 0.993)"""
        self.alive[slot_idx] = 5.0

    def deactivate(self, slot_idx):
        """Sets alive to -5.0 (sigmoid ≈ 0.007)"""
        self.alive[slot_idx] = -5.0

    def to_json(self):
        """Serializes to dict (JSON-compatible, float32 for json safety)"""
        return {
            "P": self.P.astype(np.float32).tolist(),
            "c": self.c.astype(np.float32).tolist(),
            "alpha": self.alpha.astype(np.float32).tolist(),
            "alive": self.alive.astype(np.float32).tolist(),
            "csg": self.csg.tolist(),
            "z": self.z.tolist(),
        }

    @classmethod
    def from_json(cls, data):
        """Deserializes, cast back to float16"""
        n_slots = len(data["alive"])
        n_ctrl_pts = len(data["P"][0]) if n_slots > 0 else N_CTRL_PTS
        obj = cls(n_slots=n_slots, n_ctrl_pts=n_ctrl_pts)
        if n_slots > 0:
            obj.P = np.array(data["P"], dtype=np.float16)
            obj.c = np.array(data["c"], dtype=np.float16)
            obj.alpha = np.array(data["alpha"], dtype=np.float16)
            obj.alive = np.array(data["alive"], dtype=np.float16)
            obj.csg = np.array(data["csg"], dtype=bool)
            obj.z = np.array(data["z"], dtype=np.int8)
        return obj

    def to_binary(self):
        """Serializes to compact bytes (numpy .npy format zip via np.savez)"""
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            P=self.P,
            c=self.c,
            alpha=self.alpha,
            alive=self.alive,
            csg=self.csg,
            z=self.z,
        )
        return buffer.getvalue()

    @classmethod
    def from_binary(cls, data):
        """Deserializes from compact bytes"""
        buffer = io.BytesIO(data)
        loaded = np.load(buffer)
        n_slots = len(loaded["alive"])
        n_ctrl_pts = loaded["P"].shape[1] if n_slots > 0 else N_CTRL_PTS
        obj = cls(n_slots=n_slots, n_ctrl_pts=n_ctrl_pts)
        obj.P = loaded["P"]
        obj.c = loaded["c"]
        obj.alpha = loaded["alpha"]
        obj.alive = loaded["alive"]
        obj.csg = loaded["csg"]
        obj.z = loaded["z"]
        return obj

    def to_torch(self):
        """Returns dict of torch tensors (float16, preserves dtypes)"""
        import torch

        return {
            "P": torch.from_numpy(self.P).clone(),
            "c": torch.from_numpy(self.c).clone(),
            "alpha": torch.from_numpy(self.alpha).clone(),
            "alive": torch.from_numpy(self.alive).clone(),
            "csg": torch.from_numpy(self.csg).clone(),
            "z": torch.from_numpy(self.z).clone(),
        }

    @classmethod
    def from_torch(cls, tensor_dict):
        """Deserializes from torch tensor dict"""
        obj = cls(
            n_slots=tensor_dict["alive"].shape[0],
            n_ctrl_pts=tensor_dict["P"].shape[1]
            if tensor_dict["P"].shape[0] > 0
            else N_CTRL_PTS,
        )
        obj.P = tensor_dict["P"].cpu().numpy().astype(np.float16)
        obj.c = tensor_dict["c"].cpu().numpy().astype(np.float16)
        obj.alpha = tensor_dict["alpha"].cpu().numpy().astype(np.float16)
        obj.alive = tensor_dict["alive"].cpu().numpy().astype(np.float16)
        obj.csg = tensor_dict["csg"].cpu().numpy().astype(bool)
        obj.z = tensor_dict["z"].cpu().numpy().astype(np.int8)
        return obj

    def clone(self):
        """Deep copy"""
        obj = self.__class__(n_slots=self.P.shape[0], n_ctrl_pts=self.P.shape[1])
        obj.P = self.P.copy()
        obj.c = self.c.copy()
        obj.alpha = self.alpha.copy()
        obj.alive = self.alive.copy()
        obj.csg = self.csg.copy()
        obj.z = self.z.copy()
        return obj

    def __repr__(self):
        """Shows active slot count and memory usage"""
        active_count = len(self.active_slots())
        mem_bytes = (
            self.P.nbytes
            + self.c.nbytes
            + self.alpha.nbytes
            + self.alive.nbytes
            + self.csg.nbytes
            + self.z.nbytes
        )
        return f"<CRFTensor: {active_count}/{self.P.shape[0]} active slots, {mem_bytes / 1024:.2f} KB>"


class CRFSequence:
    def __init__(self, crf_list=None, dp_dt=None):
        self.frames = crf_list if crf_list is not None else []
        self.dp_dt = dp_dt if dp_dt is not None else np.zeros((0,))

    def to_hdf5(self, path):
        """Saves the sequence to an HDF5 file"""
        with h5py.File(path, "w") as f:
            n_frames = len(self.frames)
            if n_frames == 0:
                f.create_dataset("P", shape=(0,))
                f.create_dataset("c", shape=(0,))
                f.create_dataset("alpha", shape=(0,))
                f.create_dataset("alive", shape=(0,))
                f.create_dataset("csg", shape=(0,))
                f.create_dataset("z", shape=(0,))
                f.create_dataset("dp_dt", shape=(0,))
                return

            n_slots = self.frames[0].P.shape[0]
            n_ctrl = self.frames[0].P.shape[1]

            P_all = np.zeros((n_frames, n_slots, n_ctrl, 2), dtype=np.float16)
            c_all = np.zeros((n_frames, n_slots, 3), dtype=np.float16)
            alpha_all = np.zeros((n_frames, n_slots), dtype=np.float16)
            alive_all = np.zeros((n_frames, n_slots), dtype=np.float16)
            csg_all = np.zeros((n_frames, n_slots), dtype=bool)
            z_all = np.zeros((n_frames, n_slots), dtype=np.int8)

            for i, frame in enumerate(self.frames):
                P_all[i] = frame.P
                c_all[i] = frame.c
                alpha_all[i] = frame.alpha
                alive_all[i] = frame.alive
                csg_all[i] = frame.csg
                z_all[i] = frame.z

            f.create_dataset("P", data=P_all, compression="gzip")
            f.create_dataset("c", data=c_all, compression="gzip")
            f.create_dataset("alpha", data=alpha_all, compression="gzip")
            f.create_dataset("alive", data=alive_all, compression="gzip")
            f.create_dataset("csg", data=csg_all, compression="gzip")
            f.create_dataset("z", data=z_all, compression="gzip")
            f.create_dataset("dp_dt", data=self.dp_dt, compression="gzip")

    @classmethod
    def from_hdf5(cls, path):
        """Loads a sequence from an HDF5 file"""
        frames = []
        with h5py.File(path, "r") as f:
            if "P" not in f or f["P"].size == 0:
                return cls()

            P_all = f["P"][:]
            c_all = f["c"][:]
            alpha_all = f["alpha"][:]
            alive_all = f["alive"][:]
            csg_all = f["csg"][:]
            z_all = f["z"][:]
            dp_dt = f["dp_dt"][:]

            n_frames = P_all.shape[0]
            for i in range(n_frames):
                frame = CRFTensor(n_slots=P_all.shape[1], n_ctrl_pts=P_all.shape[2])
                frame.P = P_all[i]
                frame.c = c_all[i]
                frame.alpha = alpha_all[i]
                frame.alive = alive_all[i]
                frame.csg = csg_all[i]
                frame.z = z_all[i]
                frames.append(frame)

        return cls(crf_list=frames, dp_dt=dp_dt)

    def frame(self, t):
        """Returns the CRFTensor at frame index t"""
        if 0 <= t < len(self.frames):
            return self.frames[t]
        raise IndexError(f"Frame index {t} out of bounds")

    def velocity(self, t):
        """Returns the ground truth velocity (dP_dt) at index t"""
        if self.dp_dt is not None and 0 <= t < len(self.dp_dt):
            return self.dp_dt[t]
        raise IndexError(f"Velocity index {t} out of bounds")
