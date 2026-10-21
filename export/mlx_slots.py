"""Temp-slot reuse for the MLX delegate of ExecuTorch 1.3.1 (EWP4 of EXPORTING_WORK_PACKAGES.md).

The MLX program builder recycles the index of a temporary tensor once its last reader is built ("delete-as-you-go"),
but in 1.3.1 the final numbering (MLXProgramBuilder._create_slot_mappings) gives every Slot object its own runtime
id, so no slot is ever reused. The runtime keeps every tensor of a call in its slot until the call ends: `sing`
then holds all its intermediates, about 51 GB on the M4 Max (more than its 48 GB), and swaps. Upstream ExecuTorch
fixed this after 1.3.1 by coalescing slots that share (id_space, idx) into one runtime id; apply() installs
that fix (the two methods as upstream has them) on the builder of this process, when the installed one lacks
it. It changes which slot holds a tensor, not what is computed. ExecuTorch 1.5.1 (neural_tilde 0.1.0) has the fix,
so there apply() does nothing.

    from export import mlx_slots
    mlx_slots.apply()   # before lowering
"""
from collections import defaultdict
import inspect
import logging

logger = logging.getLogger(__name__)


def _collect_used_slots(self):
    from executorch.backends.mlx.builder.slot_manager import IdSpace, IdType
    instruction_referenced = {slot for _, slot in self._tid_slot_map}
    instruction_referenced.update({slot for _, slot in self._vid_slot_map})
    used_slots = set()
    for _n, slot in self.slot_manager.name_to_slot.items():
        if not isinstance(slot, tuple):
            slot = (slot,)
        for s in slot:
            if s.id_space in (IdSpace.Constant, IdSpace.Temp):
                if s in instruction_referenced:
                    used_slots.add(s)
            else:
                used_slots.add(s)
    # distinct physical slots: those sharing (id_space, idx) are one slot reused over disjoint lifetimes
    num_tensors, num_values = defaultdict(int), defaultdict(int)
    seen = set()
    for s in used_slots:
        is_tensor = s.id_type == IdType.Tensor
        key = (is_tensor, s.id_space, s.idx)
        if key in seen:
            continue
        seen.add(key)
        if is_tensor:
            num_tensors[s.id_space] += 1
        else:
            num_values[s.id_space] += 1
    return used_slots, num_tensors, num_values


def _create_slot_mappings(self, used_slots):
    from executorch.backends.mlx.builder.slot_manager import IdSpace, IdType
    order = {IdSpace.Constant: 0, IdSpace.Input: 1, IdSpace.Output: 2, IdSpace.MutableBuffer: 3, IdSpace.Temp: 4}

    def coalesce(slots):
        mapping, key_to_global = {}, {}
        for s in sorted(slots, key=lambda s: (order[s.id_space], s.idx)):
            key = (s.id_space, s.idx)
            gid = key_to_global.get(key)
            if gid is None:
                gid = key_to_global[key] = len(key_to_global)
            mapping[s] = gid
        return mapping

    slot_to_tid = coalesce([s for s in used_slots if s.id_type == IdType.Tensor])
    slot_to_vid = coalesce([s for s in used_slots if s.id_type != IdType.Tensor])
    for tid, slot in getattr(self, '_tid_slot_map', ()):
        if slot in slot_to_tid:
            tid.idx = slot_to_tid[slot]
        else:
            logger.warning(f'Slot {slot} not found in slot_to_tid mapping')
    for vid, slot in getattr(self, '_vid_slot_map', ()):
        if slot in slot_to_vid:
            vid.idx = slot_to_vid[slot]
        else:
            logger.warning(f'Slot {slot} not found in slot_to_vid mapping')
    return slot_to_tid, slot_to_vid


def apply():
    """Installs the fix on MLXProgramBuilder unless it is there already. Returns True if it was installed."""
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
    if 'coalesce' in inspect.getsource(MLXProgramBuilder._create_slot_mappings).lower():
        return False
    MLXProgramBuilder._collect_used_slots = _collect_used_slots
    MLXProgramBuilder._create_slot_mappings = _create_slot_mappings
    return True
