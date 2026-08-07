import numpy as np
import mujoco

def _d(deg):
    """Degrés → radians."""
    return float(np.radians(deg))


HUMAN_JOINT_LIMITS = {
    # ── HANCHE GAUCHE ──────────────────────────────────────
    "left_hip_pitch_joint"      : (_d(-120), _d(20)),
    "left_hip_roll_joint"       : (_d(-25),  _d(45)),
    "left_hip_yaw_joint"        : (_d(-45),  _d(45)),
    # ── GENOU GAUCHE ───────────────────────────────────────
    "left_knee_joint"           : (_d(0),    _d(130)),
    # ── CHEVILLE GAUCHE ────────────────────────────────────
    "left_ankle_pitch_joint"    : (_d(-45),  _d(25)),
    "left_ankle_roll_joint"     : (_d(-15),  _d(15)),
    # ── HANCHE DROITE ──────────────────────────────────────
    "right_hip_pitch_joint"     : (_d(-120), _d(20)),
    "right_hip_roll_joint"      : (_d(-45),  _d(25)),
    "right_hip_yaw_joint"       : (_d(-45),  _d(45)),
    # ── GENOU DROIT ────────────────────────────────────────
    "right_knee_joint"          : (_d(0),    _d(130)),
    # ── CHEVILLE DROITE ────────────────────────────────────
    "right_ankle_pitch_joint"   : (_d(-45),  _d(25)),
    "right_ankle_roll_joint"    : (_d(-15),  _d(15)),
    # ── TORSE / TAILLE ─────────────────────────────────────
    "waist_yaw_joint"           : (_d(-45),  _d(45)),
    "waist_roll_joint"          : (_d(-29),  _d(29)),
    "waist_pitch_joint"         : (_d(-25),  _d(29)),
    # ── ÉPAULE GAUCHE ──────────────────────────────────────
    "left_shoulder_pitch_joint" : (_d(-60),  _d(153)),
    "left_shoulder_roll_joint"  : (_d(-30),  _d(90)),
    "left_shoulder_yaw_joint"   : (_d(-90),  _d(90)),
    # ── COUDE GAUCHE ───────────────────────────────────────
    "left_elbow_joint"          : (_d(0),    _d(120)),
    # ── POIGNET GAUCHE ─────────────────────────────────────
    "left_wrist_roll_joint"     : (_d(-80),  _d(80)),
    "left_wrist_pitch_joint"    : (_d(-80),  _d(80)),
    "left_wrist_yaw_joint"      : (_d(-30),  _d(30)),
    # ── ÉPAULE DROITE ──────────────────────────────────────
    "right_shoulder_pitch_joint": (_d(-60),  _d(153)),
    "right_shoulder_roll_joint" : (_d(-90),  _d(30)),
    "right_shoulder_yaw_joint"  : (_d(-90),  _d(90)),
    # ── COUDE DROIT ────────────────────────────────────────
    "right_elbow_joint"         : (_d(0),    _d(120)),
    # ── POIGNET DROIT ──────────────────────────────────────
    "right_wrist_roll_joint"    : (_d(-80),  _d(80)),
    "right_wrist_pitch_joint"   : (_d(-80),  _d(80)),
    "right_wrist_yaw_joint"     : (_d(-30),  _d(30)),
}


LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
TORSO_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

KINEMATIC_CHAIN_MAP = {
    "imu_in_pelvis": None, 
    "left_foot"    : LEFT_LEG_JOINTS + TORSO_JOINTS,
    "right_foot"   : RIGHT_LEG_JOINTS + TORSO_JOINTS,
    "imu_in_torso" : TORSO_JOINTS,
    "left_hand"    : LEFT_ARM_JOINTS + TORSO_JOINTS,
    "right_hand"   : RIGHT_ARM_JOINTS + TORSO_JOINTS,
}


PROXIMAL_WEIGHTS = {
    # ── Torse ──────────────────────────
    "waist_yaw_joint"            : 1.0,
    "waist_roll_joint"           : 1.0,
    "waist_pitch_joint"          : 1.0,
    # ── Hanches ────────────────────────
    "left_hip_pitch_joint"       : 1.0,
    "left_hip_roll_joint"        : 1.0,
    "left_hip_yaw_joint"         : 1.0,
    "right_hip_pitch_joint"      : 1.0,
    "right_hip_roll_joint"       : 1.0,
    "right_hip_yaw_joint"        : 1.0,
    # ── Épaules ────────────────────────
    "left_shoulder_pitch_joint"  : 1.0,
    "left_shoulder_roll_joint"   : 1.0,
    "left_shoulder_yaw_joint"    : 1.0,
    "right_shoulder_pitch_joint" : 1.0,
    "right_shoulder_roll_joint"  : 1.0,
    "right_shoulder_yaw_joint"   : 1.0,
    # ── Genoux / Coudes ────────────────
    "left_knee_joint"            : 0.7,
    "right_knee_joint"           : 0.7,
    "left_elbow_joint"           : 0.7,
    "right_elbow_joint"          : 0.7,
    # ── Chevilles / Poignets ───────────
    "left_ankle_pitch_joint"     : 0.4,
    "left_ankle_roll_joint"      : 0.4,
    "right_ankle_pitch_joint"    : 0.4,
    "right_ankle_roll_joint"     : 0.4,
    "left_wrist_roll_joint"      : 0.4,
    "left_wrist_pitch_joint"     : 0.4,
    "left_wrist_yaw_joint"       : 0.4,
    "right_wrist_roll_joint"     : 0.4,
    "right_wrist_pitch_joint"    : 0.4,
    "right_wrist_yaw_joint"      : 0.4,
}

# =========================================================
# POSE INITIALE
# =========================================================

INITIAL_POSE_DEG = {
    "left_knee_joint"        : 15.0,
    "right_knee_joint"       : 15.0,
    "left_hip_pitch_joint"   : -15.0,
    "right_hip_pitch_joint"  : -15.0,
}


def _apply_initial_pose(model, q):
    for jname, deg in INITIAL_POSE_DEG.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            continue
        adr = model.jnt_qposadr[jid]
        val = np.radians(deg)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            val = float(np.clip(val, lo, hi))
        q[adr] = val


def apply_human_limits(model, verbose=True):
    applied = []
    skipped = []

    for name, (lo, hi) in HUMAN_JOINT_LIMITS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            skipped.append((name, "absent du modèle"))
            continue

        jnt_type = int(model.jnt_type[jid])
        if jnt_type not in (2, 3):
            skipped.append((name, "multi-DOF ignoré"))
            continue

        lo, hi = min(lo, hi), max(lo, hi)

        xml_lo, xml_hi = model.jnt_range[jid]
        lo = max(lo, xml_lo)
        hi = min(hi, xml_hi)

        if lo >= hi:
            skipped.append((name, f"plage nulle après clamp XML "
                                   f"[{np.degrees(xml_lo):.1f}°, "
                                   f"{np.degrees(xml_hi):.1f}°]"))
            continue

        model.jnt_range[jid]   = [lo, hi]
        model.jnt_limited[jid] = 1
        applied.append(name)

    if verbose:
        print(f"[HumanLimits] {len(applied)} joint(s) contraints :")
        for n in applied:
            lo, hi = model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            print(f"  {n:45s}  [{np.degrees(lo):+7.1f}°,  {np.degrees(hi):+7.1f}°]")
        if skipped:
            print(f"\n[HumanLimits] {len(skipped)} joint(s) ignoré(s) :")
            for n, reason in skipped:
                print(f"  {n:45s}  ({reason})")

    return applied, skipped


# =========================================================
# FORWARD KINEMATICS
# =========================================================

class Forward_Kinematics:
    def __init__(self, model, data):
        self.model = model
        self.data  = data
        self.names = [
            "imu_in_pelvis",
            "imu_in_torso",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ]
        missing = [n for n in self.names
                   if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) == -1]
        if missing:
            print(f"[FK] Sites absents du modèle (ignorés) : {missing}")
        self.names = [n for n in self.names if n not in missing]
        self.ids = {n: model.site(n).id for n in self.names}

    def compute(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return {
            n: self.data.site_xpos[i].copy()
            for n, i in self.ids.items()
        }


# =========================================================
# JACOBIAN
# =========================================================

class Jacobian:
    def __init__(self, model, data):
        self.model = model
        self.data  = data

    def compute(self, site_name):
        sid  = self.model.site(site_name).id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        return jacp


# =========================================================
# IK SOLVER
# =========================================================

class IK:
    def __init__(self, model,
                 alpha=0.5, tol=1e-4, max_iters=1000,
                 dq_limit=0.1,
                 damping=1e-4,
                 apply_human_joint_limits=True):
        self.model     = model
        self.alpha     = alpha
        self.tol       = tol
        self.max_iters = max_iters
        self.dq_limit  = dq_limit
        self.damping   = damping

        self.data = mujoco.MjData(model)
        self.fk   = Forward_Kinematics(model, self.data)
        self.jac  = Jacobian(model, self.data)

        if apply_human_joint_limits:
            print("\n[IK] Application des contraintes articulaires humaines...")
            self._applied, self._skipped = apply_human_limits(model)
            print()

        self.q = self.data.qpos.copy()
        _apply_initial_pose(model, self.q)

        self._mask_cache = {}


        self._proximal_vec = np.ones(model.nv)
        for jname, w in PROXIMAL_WEIGHTS.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                continue
            dof_adr = model.jnt_dofadr[jid]
            self._proximal_vec[dof_adr] = float(w)
        self._proximal_vec[:6] = 0.3

        self.targets = {}
        self.sites   = []
        self.weights = {}

    # ------------------------------------------------------------------
    def _get_kinematic_mask(self, site_name):
        if site_name in self._mask_cache:
            return self._mask_cache[site_name]

        allowed_names = KINEMATIC_CHAIN_MAP.get(site_name, None)

        if site_name in KINEMATIC_CHAIN_MAP and allowed_names is None:
            mask = np.ones(self.model.nv, dtype=bool)
            self._mask_cache[site_name] = mask
            return mask

        if allowed_names is None:
            mask = np.ones(self.model.nv, dtype=bool)
            mask[:6] = False
            self._mask_cache[site_name] = mask
            return mask

        mask = np.zeros(self.model.nv, dtype=bool)
        for jname in allowed_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                continue
            dof_adr = self.model.jnt_dofadr[jid]
            mask[dof_adr] = True

        self._mask_cache[site_name] = mask
        return mask

    # ------------------------------------------------------------------
    def set_targets(self, targets, weights=None):
        self.targets = targets
        self.sites   = list(targets.keys())
        self.weights = weights or {}

    # ------------------------------------------------------------------
    def step(self):
        self.data.qpos[:] = self.q
        mujoco.mj_forward(self.model, self.data)

        pelvis_id  = self.fk.ids["imu_in_pelvis"]
        pelvis_pos = self.data.site_xpos[pelvis_id].copy()
        pelvis_rot = self.data.site_xmat[pelvis_id].reshape(3, 3).copy()
        Rt = pelvis_rot.T

        e_list = []
        J_list = []
        W_diag = []

        for s in self.sites:
            sid   = self.fk.ids[s]
            cur_w = self.data.site_xpos[sid].copy()
            tgt_w = self.targets[s]

            if s == 'imu_in_pelvis' or 'imu_in_pelvis' in self.sites:
                err = (tgt_w - cur_w).reshape(3, 1)
                J_w = self.jac.compute(s)
                J_l = J_w
                if s != 'imu_in_pelvis':
                    J_l[:, :6] = 0.0
            else:
                cur_l = Rt @ (cur_w - pelvis_pos)
                tgt_l = Rt @ (tgt_w - pelvis_pos)
                err   = (tgt_l - cur_l).reshape(3, 1)
                J_w = self.jac.compute(s)
                J_l = Rt @ J_w
                J_l[:, :6] = 0.0

            w = float(self.weights.get(s, 1.0))

            mask = self._get_kinematic_mask(s)
            J_l[:, ~mask] = 0.0

            e_list.append(err)
            J_list.append(J_l)
            W_diag.extend([w, w, w])

        W    = np.diag(W_diag)
        e    = np.vstack(e_list)
        J    = np.vstack(J_list)

        WJ   = W @ J
        We   = W @ e
        JtW  = WJ.T
        JtWJ = JtW @ J
        JtWe = (JtW @ We).flatten()

        sv_min           = float(np.linalg.svd(J, compute_uv=False)[-1])
        adaptive_damping = self.damping / (sv_min + 1e-6)
        adaptive_damping = float(np.clip(adaptive_damping, 1e-6, self.damping * 10))

        dq = self.alpha * np.linalg.solve(
            JtWJ + adaptive_damping * np.eye(self.model.nv),
            JtWe,
        )

        if 'imu_in_pelvis' not in self.sites:
            dq[:6] = 0.0
        dq = np.clip(dq, -self.dq_limit, self.dq_limit)

        dq *= self._proximal_vec

        mujoco.mj_integratePos(self.model, self.q, dq, 1.0)

        return self._check()

    # ------------------------------------------------------------------
    def _check(self):
        self.data.qpos[:] = self.q
        mujoco.mj_forward(self.model, self.data)

        pelvis_id  = self.fk.ids["imu_in_pelvis"]
        pelvis_pos = self.data.site_xpos[pelvis_id].copy()
        pelvis_rot = self.data.site_xmat[pelvis_id].reshape(3, 3).copy()
        Rt         = pelvis_rot.T

        for s in self.sites:
            sid   = self.fk.ids[s]
            cur_w = self.data.site_xpos[sid].copy()
            tgt_w = self.targets[s]
            if s == 'imu_in_pelvis' or 'imu_in_pelvis' in self.sites:
                err = np.linalg.norm(tgt_w - cur_w)
            else:
                cur_l = Rt @ (cur_w - pelvis_pos)
                tgt_l = Rt @ (tgt_w - pelvis_pos)
                err   = np.linalg.norm(tgt_l - cur_l)
            tol   = getattr(self, 'tol_per_site', {}).get(s, self.tol)
            if err >= tol:
                return False
        return True

    # ------------------------------------------------------------------
    def solve(self, visualize=True):
        converged = False
        i         = 0

        if visualize:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(
                self.model, self.data
            ) as viewer:
                for i in range(self.max_iters):
                    converged = self.step()
                    viewer.sync()
                    if converged:
                        print(f"[IK] Convergé en {i + 1} itération(s).")
                        break
                else:
                    print(f"[IK] Non convergé après {self.max_iters} itérations.")
        else:
            for i in range(self.max_iters):
                converged = self.step()
                if converged:
                    print(f"[IK] Convergé en {i + 1} itération(s).")
                    break
            else:
                print(f"[IK] Non convergé après {self.max_iters} itérations.")

        return self.fk.compute(self.q), i + 1, converged