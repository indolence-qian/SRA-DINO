"""Paired inference interventions and GT-only offline diagnostics. No training."""
from collections import defaultdict

import cv2
import numpy as np
import torch

from tools.vlm_review import digest

ARMS = ('real', 'zero_text', 'branch_off', 'shuffled_text')


def shuffled_semantics(embeddings, valid, ids, seed):
    """Other-image, same-category/partition donors; preserve query coverage/roles.

    Caller supplies ONE category/partition. This function takes no labels or Base
    scores. Prefer the same tile; fall back to another valid tile of the same role.
    Missing donors stay unchanged and are explicitly counted, never invented.
    """
    embeddings, valid = np.asarray(embeddings), np.asarray(valid)
    if embeddings.shape[:3] != valid.shape or len(ids) != len(embeddings) or len(set(ids)) != len(ids):
        raise ValueError('Misaligned semantic bank or duplicate image IDs')
    result = embeddings.copy()
    pools = defaultdict(list)
    for i, tile, role in np.argwhere(valid > 0):
        pools[int(role)].append((int(i), int(tile)))
    order = sorted(range(len(ids)), key=lambda i: digest([seed, ids[i]]))
    positions = {i: p for p, i in enumerate(order)}
    audit, donors = [], []
    for i, image_id in enumerate(ids):
        used = changed = slots = 0
        others = order[positions[i]+1:] + order[:positions[i]]
        for tile, role in np.argwhere(valid[i] > 0):
            slots += 1
            choices = [(j, int(tile)) for j in others if valid[j, tile, role] > 0]
            if not choices:
                choices = [(j, t) for j, t in pools[int(role)] if j != i]
            if not choices:
                continue
            j, t = choices[0]
            result[i, tile, role] = embeddings[j, t, role]
            used += 1
            changed += int(not np.array_equal(result[i, tile, role], embeddings[i, tile, role]))
            donors.append(dict(image_id=image_id, tile=int(tile), role=int(role),
                               donor_id=ids[j], donor_tile=t))
        audit.append(dict(valid_slots=slots, donor_slots=used, changed_slots=changed,
                          unavailable_slots=slots-used))
    return result, audit, donors


def intervene(batch, arm, shuffled=None):
    """zero_text isolates text content; branch_off also removes coverage/bias."""
    result = dict(batch)
    if arm == 'zero_text':
        result['embeddings'] = torch.zeros_like(batch['embeddings'])
    elif arm == 'shuffled_text':
        if shuffled is None or shuffled.shape != batch['embeddings'].shape:
            raise ValueError('Shuffled embedding shape mismatch')
        result['embeddings'] = shuffled
    elif arm not in ('real', 'branch_off'):
        raise ValueError('Unknown intervention')
    return result, 'visual' if arm == 'branch_off' else 'vlm'


def difference(a, b, prefix, epsilon):
    values = np.abs(np.asarray(a, np.float64)-np.asarray(b, np.float64))
    return {prefix+'_abs_mean': float(values.mean()), prefix+'_abs_max': float(values.max()),
            prefix+'_changed_fraction': float(np.mean(values > epsilon))}


def pixel_regions(mask, base, prob, delta, near_radius=8):
    """All GT use is restricted to reporting, after prediction."""
    mask = np.asarray(mask, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = np.zeros_like(mask)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] <= mask.size*.001:
            small |= labels == label
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((near_radius*2+1,)*2, np.uint8)) > 0
    areas = dict(small_gt=small, large_gt=mask & ~small, near_background=dilated & ~mask,
                 far_background=~dilated, low_base_gt=mask & (base < 1e-5))
    result = []
    for region, support in areas.items():
        count = int(support.sum())
        result.append(dict(region=region, pixels=count,
                           base_sum=float(base[support].astype(np.float64).sum()),
                           probability_sum=float(prob[support].astype(np.float64).sum()),
                           probability_increase_sum=float(np.maximum(prob-base, 0)[support].astype(np.float64).sum()),
                           raw_residual_sum=float(delta[support].astype(np.float64).sum())))
    return result


def component_rows(mask, base, prob, threshold, base_threshold):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows = []
    for label in range(1, n):
        support = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        old_hit = bool(np.mean(base[support] >= .5) >= .1)
        hit = bool(np.mean(prob[support] >= .5) >= .1)
        rows.append(dict(component=label, area=area, area_fraction=area/mask.size,
                         small=area <= mask.size*.001,
                         base_mean=float(base[support].mean()),
                         base_below_1e5_fraction=float(np.mean(base[support] < 1e-5)),
                         predicted_mean=float(prob[support].mean()),
                         base_recall_at_05=float(np.mean(base[support] >= .5)),
                         recall_at_05=float(np.mean(prob[support] >= .5)),
                         recovered_at_05=hit and not old_hit, lost_at_05=old_hit and not hit,
                         base_recall_at_fpr_DIAGNOSTIC=float(np.mean(base[support] > base_threshold)),
                         recall_at_fpr_DIAGNOSTIC=float(np.mean(prob[support] > threshold))))
    return rows


def save_panel(path, image_path, mask, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert('RGB').resize((mask.shape[1], mask.shape[0])))
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, .06])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(4)]
    axes[0].imshow(rgb)
    axes[0].set_title('Query')
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('GT (diagnostic only)')
    for ax, key in zip(axes[2:6], ('base', 'real', 'zero_text', 'shuffled_text')):
        handle = ax.imshow(np.maximum(maps[key], 1e-6), cmap='magma', norm=LogNorm(1e-6, 1))
        ax.set_title(key+' probability (log scale)')
    fig.colorbar(handle, cax=fig.add_subplot(grid[0, 4]), label='Probability (all maps)')
    diffs = [maps['real']-maps['base'], maps['real']-maps['shuffled_text']]
    scale = max(1e-8, max(float(np.abs(x).max()) for x in diffs))
    for ax, value, title in zip(axes[6:], diffs, ('real - Base', 'real - shuffled')):
        handle = ax.imshow(value, cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.set_title(title+' (linear)')
    fig.colorbar(handle, cax=fig.add_subplot(grid[1, 4]), label='Probability difference')
    for ax in axes:
        ax.axis('off')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
