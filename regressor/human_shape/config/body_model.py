from typing import Tuple
from loguru import logger
from copy import deepcopy
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from .utils import Variable, Pose


@dataclass
class PCA:
    num_comps: int = 12
    flat_hand_mean: bool = False


@dataclass
class PoseWithPCA(Pose):
    pca: PCA = field(default_factory=PCA)


@dataclass
class Shape(Variable):
    num: int = 10


@dataclass
class Expression(Variable):
    num: int = 10


@dataclass
class Texture(Variable):
    dim: int = 50
    path: str = 'data/flame/texture.npz'


@dataclass
class Lighting(Variable):
    dim: int = 27
    type: str = 'sh'


@dataclass
class AbstractBodyModel:
    extra_joint_path: str = ''
    v_template_path: str = ''
    mean_pose_path: str = ''
    shape_mean_path: str = ''
    use_compressed: bool = True
    gender = 'neutral'
    learn_joint_regressor: bool = False


@dataclass
class SMPL(AbstractBodyModel):
    ext: str = 'pkl'
    use_feet_keypoints: bool = True
    use_face_keypoints: bool = True
    j14_regressor_path: str = ''
    betas: Shape = field(default_factory=Shape)
    global_rot: Pose = field(default_factory=Pose)
    body_pose: Pose = field(default_factory=Pose)
    translation: Variable = field(default_factory=Variable)
    head_verts_ids_path: str = ''


@dataclass
class SMPLH(SMPL):
    left_hand_pose: PoseWithPCA = field(default_factory=PoseWithPCA)
    right_hand_pose: PoseWithPCA = field(default_factory=PoseWithPCA)


@dataclass
class SMPLX(SMPLH):
    ext: str = 'npz'
    use_face_contour: bool = False
    expression: Expression = field(default_factory=Expression)
    jaw_pose: Pose = field(default_factory=Pose)
    leye_pose: Pose = field(default_factory=Pose)
    reye_pose: Pose = field(default_factory=Pose)
    hand_vertex_ids_path: str = ''


@dataclass
class BodyModel:
    type: str = 'smplx'
    model_folder: str = 'models'
    smpl: SMPL = field(default_factory=SMPL)
    smplh: SMPLH = field(default_factory=SMPLH)
    smplx: SMPLX = field(default_factory=SMPLX)


body_conf = OmegaConf.structured(BodyModel)