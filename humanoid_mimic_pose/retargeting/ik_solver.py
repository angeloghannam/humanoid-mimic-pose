# ik_solver.py
import numpy as np
import mujoco

class Forward_Kinematics:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)

        self.names = ['imu_in_pelvis', 'imu_in_torso', 'left_foot', 'right_foot']
        self.ids = {name: model.site(name).id for name in self.names}

    def compute(self, q):
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return {name: self.data.site_xpos[sid].copy() for name, sid in self.ids.items()}

class Jacobian:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def compute(self, site_name):
        site_id = self.model.site(site_name).id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp, jacr

class IK:
    def __init__(self, model, fk_solver, jacobian_solver,
                 alpha=0.1, tolerance=1e-4, max_iters=2000):
        self.model = model
        self.fk = fk_solver
        self.jacobian = jacobian_solver
        self.alpha = alpha
        self.tol = tolerance
        self.max_iters = max_iters

        self.q = self.fk.data.qpos.copy()

        self.model.opt.gravity[:] = 0

        self.targets = {}
        self.sites = []
        self.weights = {}
        self.dof_mask = np.ones(self.model.nv, dtype=bool)

    def set_targets(self, target_dict, weights=None):
        self.targets = target_dict
        self.sites = list(target_dict.keys())
        self.weights = weights or {}

    def _joint_dofnum(self, jid: int) -> int:
        jtype = self.model.jnt_type[jid]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        
        return 1

    def restrict_to_joints(self, joint_names, lock_base=True):
        """Autorise seulement les DoF (dans l'espace nv) appartenant aux joints donnés."""
        mask = np.zeros(self.model.nv, dtype=bool)

        for jname in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            adr = self.model.jnt_dofadr[jid]
            num = self._joint_dofnum(jid)
            mask[adr:adr + num] = True

        if lock_base and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            mask[:6] = False

        self.dof_mask = mask

    def step(self, lock_base=True, damping=1e-4):
        positions = self.fk.compute(self.q)

        errors = []
        J_stack = []
        for s in self.sites:
            w = float(self.weights.get(s, 1.0))
            err = (self.targets[s] - positions[s]).reshape(3, 1)
            jacp, _ = self.jacobian.compute(s)
            errors.append(w * err)
            J_stack.append(w * jacp)

        e = np.vstack(errors)
        J = np.vstack(J_stack)

        if np.linalg.norm(e) < self.tol:
            return True

        mask = self.dof_mask.copy()
        if lock_base:
            mask[:6] = False

        idx = np.flatnonzero(mask) 
        Jm = J[:, idx]

        JJt = Jm @ Jm.T
        dq_m = self.alpha * (Jm.T @ np.linalg.solve(JJt + damping*np.eye(JJt.shape[0]), e)).flatten()

        dq = np.zeros(self.model.nv)
        dq[idx] = dq_m

        mujoco.mj_integratePos(self.model, self.q, dq, 1.0)
        self.fk.data.qpos[:] = self.q
        mujoco.mj_forward(self.model, self.fk.data)
        return False


    def solve(self, visualize=True, lock_base=True):
        if visualize:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.fk.data) as viewer:
                for i in range(self.max_iters):
                    if self.step(lock_base=lock_base):
                        print(f"Converged in {i} iterations")
                        break
                    viewer.sync()
        else:
            for i in range(self.max_iters):
                if self.step(lock_base=lock_base):
                    print(f"Converged in {i} iterations")
                    break
        return self.get_positions()

    def get_positions(self):
        return self.fk.compute(self.q)