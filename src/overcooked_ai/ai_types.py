from __future__ import annotations
import dataclasses
import enum

import numpy as np

from overcooked_ai.common_types import Point2D
from overcooked_ai.dataset_types import BBoxAnnotation


@enum.unique
class ObjectId(enum.IntEnum):
    # Static tiles
    TileFreeSpace = 0
    TileCountertop = enum.auto()
    TileTopPass = enum.auto()
    TileBottomPass = enum.auto()
    TileTrashChute = enum.auto()
    TileCuttingBoard = enum.auto()
    TilePlateReturn = enum.auto()
    TileShrimpCrate = enum.auto()
    TileTunaCrate = enum.auto()
    # Chef states
    Chef = enum.auto()
    ChefChopping = enum.auto()
    ChefCarrying = enum.auto()
    # Foreground objects
    Plate = enum.auto()
    RawShrimp = enum.auto()
    ChoppedShrimp = enum.auto()
    ChoppedShrimpOnPlate = enum.auto()
    RawTuna = enum.auto()
    ChoppedTuna = enum.auto()
    ChoppedTunaOnPlate = enum.auto()
    # Icons
    IconShrimp = enum.auto()
    IconTuna = enum.auto()
    ProgressBar = enum.auto()
    OrderShrimp = enum.auto()
    OrderTuna = enum.auto()
    # Digits
    Digit0 = enum.auto()
    Digit1 = enum.auto()
    Digit2 = enum.auto()
    Digit3 = enum.auto()
    Digit4 = enum.auto()
    Digit5 = enum.auto()
    Digit6 = enum.auto()
    Digit7 = enum.auto()
    Digit8 = enum.auto()
    Digit9 = enum.auto()


map_from_object_id_to_short_label: dict[ObjectId, str] = {
    ObjectId.TileFreeSpace: "",
    ObjectId.TileCountertop: "CtrTp",
    ObjectId.TileTopPass: "TPass",
    ObjectId.TileBottomPass: "BPass",
    ObjectId.TileTrashChute: "TrBin",
    ObjectId.TileCuttingBoard: "CtBrd",
    ObjectId.TilePlateReturn: "PlRtn",
    ObjectId.TileShrimpCrate: "ShrCt",
    ObjectId.TileTunaCrate: "TnaCt",
    ObjectId.Chef: "Chef",
    ObjectId.ChefChopping: "CfChp",
    ObjectId.ChefCarrying: "CfCry",
    ObjectId.Plate: "Plate",
    ObjectId.RawShrimp: "RwShr",
    ObjectId.ChoppedShrimp: "ChShr",
    ObjectId.ChoppedShrimpOnPlate: "ShrPl",
    ObjectId.RawTuna: "RwTna",
    ObjectId.ChoppedTuna: "ChTna",
    ObjectId.ChoppedTunaOnPlate: "TnaPl",
    ObjectId.IconShrimp: "IcnSh",
    ObjectId.IconTuna: "IcnTn",
    ObjectId.ProgressBar: "PgrBr",
    ObjectId.OrderShrimp: "OrdSh",
    ObjectId.OrderTuna: "OrdTn",
    ObjectId.Digit0: "0",
    ObjectId.Digit1: "1",
    ObjectId.Digit2: "2",
    ObjectId.Digit3: "3",
    ObjectId.Digit4: "4",
    ObjectId.Digit5: "5",
    ObjectId.Digit6: "6",
    ObjectId.Digit7: "7",
    ObjectId.Digit8: "8",
    ObjectId.Digit9: "9",
}

map_from_short_label_to_object_id: dict[str, ObjectId] = {
    label: object_id for object_id, label in map_from_object_id_to_short_label.items()
}

IsTile: set[ObjectId] = {id for id in ObjectId.__members__.values() if id.name.startswith("Tile")}

IsChef: set[ObjectId] = {id for id in ObjectId.__members__.values() if id.name.startswith("Chef")}

IsForeground: set[ObjectId] = {
    ObjectId.Plate,
    ObjectId.RawShrimp,
    ObjectId.ChoppedShrimp,
    ObjectId.ChoppedShrimpOnPlate,
    ObjectId.RawTuna,
    ObjectId.ChoppedTuna,
    ObjectId.ChoppedTunaOnPlate,
}

IsIcon: set[ObjectId] = {
    ObjectId.IconShrimp,
    ObjectId.IconTuna,
    ObjectId.ProgressBar,
    ObjectId.OrderShrimp,
    ObjectId.OrderTuna,
}

IsDigit: set[ObjectId] = {id for id in ObjectId.__members__.values() if id.name.startswith("Digit")}


@enum.unique
class EntityId(enum.IntEnum):
    """Entities are mostly combinations of objects, towards progressing game state."""

    Chef = enum.auto()
    ChefCarryingPlate = enum.auto()
    ChefCarryingRawShrimp = enum.auto()
    ShrimpOnCuttingBoard = enum.auto()
    ChefChoppingShrimp = enum.auto()
    ChefCarryingChoppedShrimp = enum.auto()
    ChoppedShrimpOnPlate = enum.auto()
    ChefCarryingChoppedShrimpOnPlate = enum.auto()
    ChefCarryingRawTuna = enum.auto()
    TunaOnCuttingBoard = enum.auto()
    ChefChoppingTuna = enum.auto()
    ChefCarryingChoppedTuna = enum.auto()
    ChoppedTunaOnPlate = enum.auto()
    ChefCarryingChoppedTunaOnPlate = enum.auto()


@enum.unique
class ActionId(enum.IntEnum):
    GrabOrDrop = enum.auto()
    Chop = enum.auto()
    MoveUp = enum.auto()
    MoveDown = enum.auto()
    MoveLeft = enum.auto()
    MoveRight = enum.auto()
    MoveUpRight = enum.auto()
    MoveDownRight = enum.auto()
    MoveDownLeft = enum.auto()
    MoveUpLeft = enum.auto()


@dataclasses.dataclass
class MapGrid:
    map_from_row_col_to_tile_id: list[list[ObjectId]]

    @property
    def height(self) -> int:
        return len(self.map_from_row_col_to_tile_id)

    @property
    def width(self) -> int:
        return len(self.map_from_row_col_to_tile_id[0])


@dataclasses.dataclass
class RoundState:
    score: int
    secs_remaining: int


@dataclasses.dataclass
class BBoxDetections:
    map_from_object_id_to_detections: dict[ObjectId, list[BBoxAnnotation]]


@dataclasses.dataclass
class Icon:
    detection: BBoxAnnotation
    object_id: ObjectId


@dataclasses.dataclass
class Object(Icon):
    position_in_frame_space_px: Point2D
    position_in_grid_space_px: Point2D


@dataclasses.dataclass
class Entity:
    entity_id: EntityId
    components: list[Icon]


@dataclasses.dataclass
class EntitiesState:
    chef_entities: list[Entity]  # fixed size and order
    food_entities: list[Entity]  # not associated to chef
    non_associated_objects: list[Object]


@dataclasses.dataclass
class Order(Icon):
    submission_time_sec: float


@dataclasses.dataclass
class OrdersList:
    orders: list[Order]


@dataclasses.dataclass
class DishState:
    entity: Entity
    order: Order


@dataclasses.dataclass
class DishesPlan:
    map_from_chef_idx_to_dish_state: dict[int, DishState]


@dataclasses.dataclass
class SceneState:
    round: RoundState
    entities: EntitiesState
    orders: OrdersList


@dataclasses.dataclass
class PredictedSceneState(SceneState):
    stale_orders: OrdersList


@dataclasses.dataclass
class Action:
    action_id: ActionId


# @dataclasses.dataclass
# class ContinuousMoveAction(Action):
#     orientation_deg: float  # 0 = up, 90 = right, 180 = down, 270 = left


@dataclasses.dataclass
class ActionHistory:
    timestamped_actions: dict[float, Action]


@dataclasses.dataclass
class TrajectorySegment(Action):
    start_position_in_grid_space_px: Point2D
    end_position_in_grid_space_px: Point2D


@dataclasses.dataclass
class Trajectory:
    target_dish: DishState
    actions: list[Action]


@dataclasses.dataclass
class AgentState(SceneState):
    dishes: DishesPlan
    frame: np.ndarray
    detections: BBoxDetections
